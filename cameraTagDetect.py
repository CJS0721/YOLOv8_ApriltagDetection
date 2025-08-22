#!/usr/bin/env python3
import cv2
import time
import threading
import queue
from ultralytics import YOLO

# 直接在代码中设置模型路径和摄像头设备
MODEL_PATH = "runs/detect/train3/weights/best.pt"
VIDEO_DEVICE = "/dev/video2"
CONFIDENCE = 0.5

# 摄像头参数设置 - 设置为0表示使用默认值
CAMERA_FPS = 30
CAMERA_WIDTH = 0   # 0 = 使用默认宽度
CAMERA_HEIGHT = 0  # 0 = 使用默认高度

class VideoStreamProcessor:
    def __init__(self, model_path, video_device, confidence=0.5, max_queue_size=5, 
                 camera_fps=30, camera_width=0, camera_height=0):
        self.model = YOLO(model_path)
        self.video_device = video_device
        self.confidence = confidence
        self.camera_fps = camera_fps
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.running = False
        self.fps_counter = FPSCounter()
        
    def detect_camera_capabilities(self):
        """检测摄像头能力"""
        cap = cv2.VideoCapture(self.video_device)
        if not cap.isOpened():
            print(f"错误: 无法打开视频设备 {self.video_device}")
            return None
            
        # 获取默认分辨率
        default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        default_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n=== 摄像头信息 ===")
        print(f"设备: {self.video_device}")
        print(f"默认分辨率: {default_width} x {default_height}")
        print(f"默认帧率: {default_fps}")
        
        # 测试常见分辨率
        common_resolutions = [
            (320, 240),   # QVGA
            (640, 480),   # VGA
            (800, 600),   # SVGA
            (1024, 768),  # XGA
            (1280, 720),  # HD 720p
            (1280, 960),  # SXGA
            (1920, 1080), # Full HD 1080p
        ]
        
        supported_resolutions = []
        print(f"\n=== 支持的分辨率测试 ===")
        
        for width, height in common_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                supported_resolutions.append((width, height))
                aspect_ratio = width / height
                print(f"✓ {width}x{height} (比例 {aspect_ratio:.2f}:1)")
            else:
                print(f"✗ {width}x{height} (不支持)")
        
        cap.release()
        
        return {
            'default': (default_width, default_height),
            'default_fps': default_fps,
            'supported': supported_resolutions
        }
        
    def start(self):
        """启动视频处理线程"""
        # 首先检测摄像头能力
        camera_info = self.detect_camera_capabilities()
        if camera_info is None:
            return
            
        # 如果没有设置分辨率，使用默认值
        if self.camera_width == 0 or self.camera_height == 0:
            self.camera_width, self.camera_height = camera_info['default']
            print(f"\n使用默认分辨率: {self.camera_width}x{self.camera_height}")
        
        self.running = True
        
        # 启动各个线程
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self.display_thread = threading.Thread(target=self.display_frames)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        self.display_thread.join()
        
    def capture_frames(self):
        """从摄像头捕获帧并放入队列"""
        cap = cv2.VideoCapture(self.video_device)
        if not cap.isOpened():
            print(f"错误: 无法打开视频设备 {self.video_device}")
            self.running = False
            return
        
        # 设置摄像头参数
        if self.camera_width > 0 and self.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        
        # 获取实际参数
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = actual_width / actual_height
        
        print(f"\n=== 当前使用参数 ===")
        print(f"分辨率: {actual_width}x{actual_height}")
        print(f"画幅比例: {aspect_ratio:.2f}:1")
        print(f"帧率: {actual_fps} fps")
        print("实时检测已启动，按ESC键退出...")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法从摄像头读取帧")
                self.running = False
                break
                
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put(frame)
            
        cap.release()
        
    def process_frames(self):
        """处理队列中的帧"""
        while self.running:
            try:
                frame = self.frame_queue.get_nowait()
                results = self.model(frame, conf=self.confidence)
                self.result_queue.put((frame, results[0]))
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"处理帧时出错: {e}")
                
    def display_frames(self):
        """显示处理后的帧"""
        last_frame = None
        
        while self.running:
            try:
                frame, result = self.result_queue.get_nowait()
                
                annotated_frame = result.plot()
                last_frame = annotated_frame
                
                # 添加信息显示
                self.fps_counter.update()
                fps_text = f"Processing FPS: {self.fps_counter.get_fps():.1f}"
                resolution_text = f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
                
                cv2.putText(annotated_frame, fps_text, (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, resolution_text, (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow("AprilTag Detection", annotated_frame)
                
            except queue.Empty:
                if last_frame is not None:
                    cv2.imshow("AprilTag Detection", last_frame)
                time.sleep(0.01)
            except Exception as e:
                print(f"显示帧时出错: {e}")
                
            key = cv2.waitKey(1)
            if key == 27:  # ESC键
                self.running = False
                break
                
        cv2.destroyAllWindows()

class FPSCounter:
    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.frame_times = []
        
    def update(self):
        self.frame_times.append(time.time())
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
            
    def get_fps(self):
        if len(self.frame_times) <= 1:
            return 0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0

def main():
    print(f"加载模型: {MODEL_PATH}")
    print(f"连接摄像头: {VIDEO_DEVICE}")
    
    processor = VideoStreamProcessor(
        model_path=MODEL_PATH,
        video_device=VIDEO_DEVICE,
        confidence=CONFIDENCE,
        camera_fps=CAMERA_FPS,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT
    )
    
    try:
        processor.start()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        processor.running = False
        print("程序已退出")

if __name__ == "__main__":
    main()
