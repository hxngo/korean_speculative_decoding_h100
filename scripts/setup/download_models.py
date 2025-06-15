# download_models.py - H100 연구용 필수 모델 다운로드
import os
import time
import json
from datetime import datetime
import argparse
from huggingface_hub import snapshot_download
import shutil

# H100 연구에 필수적인 모델들
ESSENTIAL_MODELS = {
    # 1. M3 성능 역전 현상 검증용 (크기별 비교)
    "size_comparison": {
        "models": [
            ("microsoft/DialoGPT-small", "117M", "M3 실험 기준점"),
            ("Qwen/Qwen2-1.5B-Instruct", "1.5B", "M3 1.3B 대체"),
            ("microsoft/Phi-3.5-mini-instruct", "3.8B", "M3 3.8B 직접 비교"),
            ("Qwen/Qwen2-7B-Instruct", "7B", "중간 크기 기준"),
            ("microsoft/Phi-3-small-8k-instruct", "7B", "대안 7B 모델"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "7B", "Mistral 7B"),
        ],
        "priority": 1,
        "description": "M3 성능 역전 현상을 H100에서 검증하기 위한 크기별 모델"
    },
    
    # 2. 최신 추론 특화 모델들
    "reasoning_models": {
        "models": [
            ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "7B", "DeepSeek R1 경량화"),
            ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "14B", "DeepSeek R1 중형"),
            ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "32B", "DeepSeek R1 대형"),
            ("Qwen/QwQ-32B-Preview", "32B", "Qwen 추론 특화"),
            ("Qwen/Qwen2.5-72B-Instruct", "72B", "Qwen 최신 대형"),
        ],
        "priority": 2,
        "description": "2025년 최신 추론 능력 강화 모델들"
    },
    
    # 3. 한국어 특화 모델들
    "korean_models": {
        "models": [
            ("upstage/SOLAR-10.7B-Instruct-v1.0", "10.7B", "세계 1위 한국 모델"),
            ("nlpai-lab/kullm-polyglot-12.8b-v2", "12.8B", "고려대 한국어 특화"),
            ("beomi/llama-2-ko-7b", "7B", "한국어 Llama"),
            ("kakaobrain/kogpt", "6B", "카카오브레인 모델"),
        ],
        "priority": 3,
        "description": "한국어 성능 최적화 모델들"
    },
    
    # 4. 벤치마크 리더 모델들
    "benchmark_leaders": {
        "models": [
            ("meta-llama/Llama-3.3-70B-Instruct", "70B", "최신 Llama"),
            ("microsoft/Phi-4", "14B", "최신 Phi"),
            ("google/gemma-2-27b-it", "27B", "Gemma 대형"),
            ("mistralai/Mistral-Nemo-Instruct-2407", "12B", "Mistral 최신"),
        ],
        "priority": 4,
        "description": "2025년 벤치마크 리더보드 상위 모델들"
    }
}

class H100ModelDownloader:
    def __init__(self, cache_dir="../../models"):
        self.cache_dir = os.path.abspath(cache_dir)
        self.log_file = os.path.join(self.cache_dir, "download_log.json")
        self.download_log = self.load_log()
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_log(self):
        """다운로드 로그 로드"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"downloads": {}, "metadata": {"created": datetime.now().isoformat()}}
    
    def save_log(self):
        """로그 저장"""
        self.download_log["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.log_file, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def check_disk_space(self, required_gb=50):
        """디스크 공간 확인"""
        free_space = shutil.disk_usage(self.cache_dir).free / (1024**3)
        return free_space, free_space > required_gb
    
    def estimate_total_size(self, category_name):
        """카테고리별 총 크기 추정"""
        if category_name not in ESSENTIAL_MODELS:
            return 0
            
        size_map = {
            "117M": 0.5, "1.5B": 6, "3.8B": 15, "6B": 24, "7B": 28,
            "9B": 36, "10.7B": 42, "12B": 48, "12.8B": 51, "14B": 56,
            "27B": 108, "32B": 128, "70B": 280, "72B": 288
        }
        
        total_gb = 0
        for model_name, size, desc in ESSENTIAL_MODELS[category_name]["models"]:
            total_gb += size_map.get(size, 20)  # 기본값 20GB
        return total_gb
    
    def download_model(self, model_name, size, description):
        """개별 모델 다운로드"""
        print(f"\n{'='*60}")
        print(f"📥 Downloading: {model_name}")
        print(f"📊 Size: {size} | 📝 {description}")
        print(f"{'='*60}")
        
        # 이미 다운로드된 경우
        if model_name in self.download_log["downloads"]:
            status = self.download_log["downloads"][model_name].get("status")
            if status == "success":
                print(f"✅ Already downloaded: {model_name}")
                return True
        
        try:
            start_time = time.time()
            
            # HuggingFace Hub로 다운로드
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir,
                local_files_only=False,
                resume_download=True,
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 로그 업데이트
            self.download_log["downloads"][model_name] = {
                "status": "success",
                "download_time": datetime.now().isoformat(),
                "duration_seconds": round(duration, 1),
                "size": size,
                "description": description,
                "local_path": local_path
            }
            
            print(f"✅ Success: {model_name}")
            print(f"⏱️  Download time: {duration:.1f}s")
            print(f"📁 Path: {local_path}")
            
            self.save_log()
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            
            self.download_log["downloads"][model_name] = {
                "status": "failed",
                "error": str(e),
                "download_time": datetime.now().isoformat()
            }
            self.save_log()
            return False
    
    def download_category(self, category_name, ask_confirmation=True):
        """카테고리별 다운로드"""
        if category_name not in ESSENTIAL_MODELS:
            print(f"❌ Unknown category: {category_name}")
            print(f"Available: {list(ESSENTIAL_MODELS.keys())}")
            return
        
        category = ESSENTIAL_MODELS[category_name]
        models = category["models"]
        
        print(f"\n🎯 Category: {category_name}")
        print(f"📋 Description: {category['description']}")
        print(f"📊 Models: {len(models)}")
        
        # 크기 추정
        total_size = self.estimate_total_size(category_name)
        print(f"💾 Estimated size: {total_size:.1f} GB")
        
        # 디스크 공간 확인
        free_space, has_space = self.check_disk_space(total_size)
        print(f"💿 Available space: {free_space:.1f} GB")
        
        if not has_space:
            print(f"⚠️  Warning: Insufficient disk space!")
            
        # 모델 목록 표시
        print(f"\n📋 Models to download:")
        for i, (model_name, size, desc) in enumerate(models, 1):
            status = "✅" if model_name in self.download_log["downloads"] else "⏳"
            print(f"   {i}. {status} {model_name} ({size}) - {desc}")
        
        # 확인
        if ask_confirmation:
            response = input(f"\n🚀 Start download? (y/N): ")
            if response.lower() != 'y':
                print("❌ Download cancelled")
                return
        
        # 다운로드 실행
        print(f"\n🚀 Starting downloads...")
        success_count = 0
        
        for i, (model_name, size, desc) in enumerate(models, 1):
            print(f"\n📦 [{i}/{len(models)}] Processing...")
            if self.download_model(model_name, size, desc):
                success_count += 1
            
            # 간단한 대기
            time.sleep(1)
        
        print(f"\n🎉 Category '{category_name}' download complete!")
        print(f"✅ Success: {success_count}/{len(models)} models")
        
    def show_categories(self):
        """사용 가능한 카테고리 표시"""
        print("\n📋 Available Categories (Priority Order):")
        print("=" * 60)
        
        # 우선순위별 정렬
        sorted_categories = sorted(
            ESSENTIAL_MODELS.items(), 
            key=lambda x: x[1]["priority"]
        )
        
        for name, info in sorted_categories:
            size = self.estimate_total_size(name)
            model_count = len(info["models"])
            priority = info["priority"]
            
            print(f"\n🏷️  {name} (Priority: {priority})")
            print(f"   📝 {info['description']}")
            print(f"   📊 Models: {model_count}")
            print(f"   💾 Est. Size: {size:.1f} GB")
    
    def show_status(self):
        """다운로드 상태 표시"""
        downloads = self.download_log["downloads"]
        
        if not downloads:
            print("📭 No downloads yet")
            return
        
        print(f"\n📊 Download Status ({len(downloads)} models)")
        print("=" * 60)
        
        success_count = sum(1 for d in downloads.values() if d.get("status") == "success")
        failed_count = len(downloads) - success_count
        
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        
        # 카테고리별 상태
        for category_name, category_info in ESSENTIAL_MODELS.items():
            models_in_category = [m[0] for m in category_info["models"]]
            downloaded = [m for m in models_in_category if m in downloads and downloads[m].get("status") == "success"]
            
            print(f"\n📂 {category_name}: {len(downloaded)}/{len(models_in_category)}")
            for model in models_in_category:
                if model in downloads:
                    status = "✅" if downloads[model].get("status") == "success" else "❌"
                    print(f"   {status} {model}")
                else:
                    print(f"   ⏳ {model}")

def main():
    parser = argparse.ArgumentParser(description="H100 Essential Model Downloader")
    parser.add_argument("--category", type=str, help="Download specific category")
    parser.add_argument("--list", action="store_true", help="List available categories")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--all", action="store_true", help="Download all categories")
    parser.add_argument("--cache-dir", type=str, default="../../models", help="Cache directory")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    downloader = H100ModelDownloader(cache_dir=args.cache_dir)
    
    if args.list:
        downloader.show_categories()
    elif args.status:
        downloader.show_status()
    elif args.all:
        # 우선순위 순으로 모든 카테고리 다운로드
        categories = sorted(ESSENTIAL_MODELS.keys(), 
                          key=lambda x: ESSENTIAL_MODELS[x]["priority"])
        for category in categories:
            downloader.download_category(category, not args.yes)
    elif args.category:
        downloader.download_category(args.category, not args.yes)
    else:
        print("🚀 H100 Essential Model Downloader")
        print("=" * 50)
        print("\nUsage examples:")
        print("  python download_models.py --list")
        print("  python download_models.py --category size_comparison")
        print("  python download_models.py --category reasoning_models --yes")
        print("  python download_models.py --all")
        print("  python download_models.py --status")
        print("\nRecommended order:")
        print("  1. size_comparison (M3 비교용)")
        print("  2. reasoning_models (최신 추론)")
        print("  3. korean_models (한국어 특화)")
        print("  4. benchmark_leaders (성능 리더)")

if __name__ == "__main__":
    # 필수 패키지 확인
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("📦 Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
    
    main()