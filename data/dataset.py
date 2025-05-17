import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
from facenet_pytorch import MTCNN
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from data.preprocessing import FacePreprocessor

# Define function at module level for multiprocessing
def process_video_standalone(work_item):
    idx, video_path, cache_path, num_frames = work_item
    
    try:
        # Create a temporary MTCNN detector for this process
        face_detector = MTCNN(keep_all=False, device='cpu')
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return f"Failed (no frames): {video_path}"
            
        step = max(1, total_frames // num_frames)
        frames = []
        frame_idx = 0
        
        while len(frames) < num_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            try:
                boxes, _ = face_detector.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    # Get the first face
                    box = boxes[0].astype(int)
                    x1, y1, x2, y2 = box
                    
                    # Add some margin
                    h, w = frame.shape[:2]
                    margin = int(min(w, h) * 0.1)
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)
                    
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        frames.append(face)
            except Exception as e:
                print(f"Error detecting face: {e}")
                
            frame_idx += step
            
        cap.release()
        
        # If we couldn't extract enough faces, pad with the last face
        if frames and len(frames) < num_frames:
            last_face = frames[-1]
            frames.extend([last_face] * (num_frames - len(frames)))
        
        # Save to cache
        if frames and len(frames) > 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(frames, f)
            return f"Processed: {video_path}"
        else:
            return f"Failed (no frames): {video_path}"
    except Exception as e:
        return f"Error: {video_path} - {str(e)}"

class DeepfakeDataset(Dataset):
    def __init__(self, data_path, transform=None, mode='train', annotations_path=None, preprocess=False, num_frames=6, cache_dir=None):
        self.data_path = data_path
        self.mode = mode
        self.annotations_path = annotations_path
        self.num_frames = num_frames
        self.cache_dir = cache_dir or os.path.join(data_path, 'preprocessed_cache')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.samples = self._load_samples()
        self.processed_frames_cache = {}
        
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        # Run preprocessing if requested
        if preprocess:
            self._preprocess_all_videos()
            
    def _load_samples(self):
        samples = []
        
        # Load annotations from Excel file
        if self.annotations_path and os.path.exists(self.annotations_path):
            try:
                # Read Excel file
                df = pd.read_excel(self.annotations_path)
                
                print(f"Excel columns: {df.columns.tolist()}")
                print(f"First few rows: {df.head()}")
                
                # Get list of actual files in directories
                fake_dir = os.path.join(self.data_path, 'fake')
                real_dir = os.path.join(self.data_path, 'real')
                
                fake_files = os.listdir(fake_dir) if os.path.exists(fake_dir) else []
                real_files = os.listdir(real_dir) if os.path.exists(real_dir) else []
                
                print(f"Found {len(fake_files)} fake files and {len(real_files)} real files")
                
                # Direct file mapping approach
                # Process each file in the directories
                for filename in fake_files:
                    samples.append({
                        'video_path': os.path.join(fake_dir, filename),
                        'label': 1,  # 1 for fake
                        'demographics': 0  # Default to male, will be updated if possible
                    })
                    
                for filename in real_files:
                    samples.append({
                        'video_path': os.path.join(real_dir, filename),
                        'label': 0,  # 0 for real
                        'demographics': 0  # Default to male, will be updated if possible
                    })
                
                # Try to update demographics from Excel if possible
                # Create a mapping of filenames to gender
                gender_map = {}
                for _, row in df.iterrows():
                    try:
                        excel_filename = str(row['Filename'])
                        if 'Gender' in df.columns:
                            gender = row['Gender']
                            if isinstance(gender, str):
                                gender_id = 0 if gender.strip().upper().startswith('M') else 1
                            else:
                                gender_id = 0 if gender == 0 else 1
                            gender_map[excel_filename] = gender_id
                    except Exception as e:
                        print(f"Error processing gender for {row}: {e}")
                
                # Update demographics in samples where possible
                for sample in samples:
                    filename = os.path.basename(sample['video_path'])
                    # Try exact match
                    if filename in gender_map:
                        sample['demographics'] = gender_map[filename]
                    else:
                        # Try to match by ID or other pattern
                        for excel_file, gender_id in gender_map.items():
                            # Extract ID from filename if possible
                            if excel_file.split('_')[0] in filename:
                                sample['demographics'] = gender_id
                                break
                
            except Exception as e:
                print(f"Error reading Excel file: {e}")
        
        # Debug information
        print(f"Found {len(samples)} samples in {self.mode} mode")
        
        # Split dataset for train/val/test if needed
        if self.mode != 'all' and len(samples) > 0:
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(len(samples))
            
            if self.mode == 'train':
                # Use 80% for training (changed from 70%)
                samples = [samples[i] for i in indices[:int(0.8 * len(samples))]]
            elif self.mode == 'val':
                # Use 10% for validation (changed from 15%)
                samples = [samples[i] for i in indices[int(0.8 * len(samples)):int(0.9 * len(samples))]]
            elif self.mode == 'test':
                # Use 10% for testing (changed from 15%)
                samples = [samples[i] for i in indices[int(0.9 * len(samples)):]]
        
        print(f"After splitting: {len(samples)} samples in {self.mode} mode")
        return samples
    
    def _get_cache_path(self, video_path):
        """Generate a unique cache file path for a video"""
        video_name = os.path.basename(video_path)
        cache_filename = f"{os.path.splitext(video_name)[0]}_frames_{self.num_frames}.pkl"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _process_video(self, sample):
        """Process a single video and save to cache"""
        video_path = sample['video_path']
        cache_path = self._get_cache_path(video_path)
        
        # Skip if already cached
        if os.path.exists(cache_path):
            return f"Skipped (cached): {video_path}"
            
        try:
            frames = self._extract_frames(video_path, self.num_frames)
            if frames is not None and len(frames) > 0:
                with open(cache_path, 'wb') as f:
                    pickle.dump(frames, f)
                return f"Processed: {video_path}"
            else:
                return f"Failed (no frames): {video_path}"
        except Exception as e:
            return f"Error: {video_path} - {str(e)}"
    
    def _preprocess_all_videos(self):
        """Preprocess all videos in advance and save to cache"""
        print(f"Preprocessing {len(self.samples)} videos for {self.mode} set...")
        
        # Process videos sequentially if there are few samples or in parallel otherwise
        if len(self.samples) < 10:  # Small dataset - process sequentially
            results = []
            for sample in tqdm(self.samples, desc="Preprocessing videos"):
                results.append(self._process_video(sample))
        else:
            # Use a different approach that doesn't require pickling instance methods
            video_paths = [sample['video_path'] for sample in self.samples]
            cache_paths = [self._get_cache_path(path) for path in video_paths]
            
            # Create work items that don't reference self
            work_items = []
            results = []
            
            for i, (video_path, cache_path) in enumerate(zip(video_paths, cache_paths)):
                if os.path.exists(cache_path):
                    results.append(f"Skipped (cached): {video_path}")
                else:
                    # Only process uncached videos
                    work_items.append((i, video_path, cache_path, self.num_frames))
            
            # Process videos in parallel
            if work_items:
                num_workers = min(cpu_count(), 8)  # Use up to 8 cores
                with Pool(num_workers) as pool:
                    parallel_results = list(tqdm(
                        pool.imap(process_video_standalone, work_items),
                        total=len(work_items),
                        desc="Preprocessing videos"
                    ))
                results.extend(parallel_results)
        
        # Report results
        success = sum(1 for r in results if r.startswith("Processed"))
        skipped = sum(1 for r in results if r.startswith("Skipped"))
        failed = sum(1 for r in results if r.startswith(("Failed", "Error")))
        
        print(f"Preprocessing complete: {success} processed, {skipped} skipped, {failed} failed")

    def _extract_frames(self, video_path, num_frames=6):
        """Extract frames from video and detect faces with fallback"""
        # Check cache first
        cache_path = self._get_cache_path(video_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache for {video_path}: {e}")
                # If cache loading fails, continue with normal extraction
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames and calculate step size
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None
            
        step = max(1, total_frames // num_frames)
        
        # Create detectors locally in this method instead of using instance variables
        face_detector = MTCNN(keep_all=False, device='cpu')
        haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        frame_idx = 0
        while len(frames) < num_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face with MTCNN
            face_detected = False
            try:
                boxes, _ = face_detector.detect(frame)  # Use local variable
                if boxes is not None and len(boxes) > 0:
                    # Get the first face
                    box = boxes[0].astype(int)
                    x1, y1, x2, y2 = box
                    
                    # Add some margin
                    h, w = frame.shape[:2]
                    margin = int(min(w, h) * 0.1)
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)
                    
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        frames.append(face)
                        face_detected = True
            except Exception as e:
                print(f"MTCNN error: {e}")
            
            # Fallback to Haar cascade if MTCNN fails
            if not face_detected:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    faces = haar_detector.detectMultiScale(gray, 1.3, 5)  # Use local variable
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        
                        # Add margin
                        h_img, w_img = frame.shape[:2]
                        margin = int(min(w_img, h_img) * 0.1)
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(w_img, x + w + margin)
                        y2 = min(h_img, y + h + margin)
                        
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            frames.append(face)
                            face_detected = True
                except Exception as e:
                    print(f"Haar cascade error: {e}")
                
            # If all detection methods fail, use the whole frame
            if not face_detected:
                # Resize to a reasonable size if necessary
                h, w = frame.shape[:2]
                if h > 300 and w > 300:
                    # Center crop
                    crop_size = min(h, w)
                    start_h = (h - crop_size) // 2
                    start_w = (w - crop_size) // 2
                    frame = frame[start_h:start_h+crop_size, start_w:start_w+crop_size]
                frames.append(frame)
            
            frame_idx += step
            
        cap.release()
        
        # If we couldn't extract enough faces, pad with the last face
        if frames and len(frames) < num_frames:
            last_face = frames[-1]
            frames.extend([last_face] * (num_frames - len(frames)))
        
        # Save to cache
        if frames and len(frames) > 0:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(frames, f)
            except Exception as e:
                print(f"Error saving cache for {video_path}: {e}")
            
        return frames
        
    def _get_default_transforms(self):
        if self.mode == 'train':
            return A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frames from video
        frames = self._extract_frames(sample['video_path'], self.num_frames)
        
        if frames is None or len(frames) == 0:
            # Fallback: create a blank image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            transformed = self.transform(image=image)
            image = transformed['image']
            face_mask = torch.zeros((1, 224, 224), dtype=torch.float32)
            
            return {
                'image': image,
                'face_mask': face_mask,
                'label': torch.tensor(sample['label'], dtype=torch.long),
                'demographics': torch.tensor(sample['demographics'], dtype=torch.long)
            }
        
        # Create a face preprocessor locally for this method
        face_preprocessor = FacePreprocessor()
        
        # Process each frame
        processed_frames = []
        face_masks = []
        for frame in frames:
            # Generate face mask for the frame
            face_mask = face_preprocessor.extract_face_mask(frame)
            
            # Apply transformations to both image and mask
            transformed = self.transform(image=frame, mask=face_mask)
            processed_frames.append(transformed['image'])
            face_masks.append(torch.tensor(transformed['mask']).unsqueeze(0))
            
        # Stack frames and masks
        stacked_frames = torch.stack(processed_frames)
        stacked_masks = torch.stack(face_masks)
        
        return {
            'image': stacked_frames,
            'face_mask': stacked_masks,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'demographics': torch.tensor(sample['demographics'], dtype=torch.long)
        }