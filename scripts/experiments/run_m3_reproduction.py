#!/usr/bin/env python3
# run_m3_reproduction.py - M3 성능 역전 현상 H100 검증 실험

import os
import json
import time
import torch
import argparse
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    GenerationConfig
)
import pandas as pd
from typing import List, Dict, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class M3ReproductionExperiment:
    """M3 성능 역전 현상을 H100에서 재현하는 실험 클래스"""
    
    def __init__(self, cache_dir="./models", output_dir="./results"):
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self.gpu_id = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")
            self.gpu_id = None

        self.models = {}
        self.tokenizers = {}
        self.results = []
        
        # H100 전용 설정
        self.h100_config = {
            "max_memory_per_gpu": "15GB",  # 모델당 최대 메모리
            "batch_size": 2,                 
            "max_new_tokens": 512,           # 긴 생성 테스트
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # M3에서 다운로드된 모델들
        self.available_models = {
            # 기본 M3 모델들
            "117M": "microsoft/DialoGPT-small",
            "1.5B": "Qwen/Qwen2-1.5B-Instruct", 
            "3.8B": "microsoft/Phi-3.5-mini-instruct",
            "7B": "Qwen/Qwen2-7B-Instruct",
            "7B_alt": "microsoft/Phi-3-small-8k-instruct",
    
            # 한국어 특화 모델들
            "beomi/llama-2-ko-7b": "beomi/llama-2-ko-7b",
            "kakaobrain/kogpt": "kakaobrain/kogpt", 
            "nlpai-lab/kullm-polyglot-12.8b-v2": "nlpai-lab/kullm-polyglot-12.8b-v2",
            "upstage/SOLAR-10.7B-Instruct-v1.0": "upstage/SOLAR-10.7B-Instruct-v1.0",
    
            # 크기별 별칭 (편의용)
            "ko-7b": "beomi/llama-2-ko-7b",
            "kogpt": "kakaobrain/kogpt",
            "kullm": "nlpai-lab/kullm-polyglot-12.8b-v2",
            "solar": "upstage/SOLAR-10.7B-Instruct-v1.0"
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model(self, model_size: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """모델과 토크나이저 로드"""
        if model_size not in self.available_models:
            raise ValueError(f"Model size {model_size} not available. Available: {list(self.available_models.keys())}")
    
        model_name = self.available_models[model_size]
    
        if model_size in self.models:
            logger.info(f"♻️  Reusing cached model: {model_name}")
            return self.tokenizers[model_size], self.models[model_size]
    
        logger.info(f"📥 Loading model: {model_name} ({model_size})")
        start_time = time.time()
    
        try:
            # KoGPT 특별 처리
            if model_name == "kakaobrain/kogpt":
                return self.load_kogpt_model(model_size)
        
            # KULLM 특별 처리
            if model_name == "nlpai-lab/kullm-polyglot-12.8b-v2":
                return self.load_kullm_model(model_size)
        
            # 일반 모델 로드
            return self.load_regular_model(model_size, model_name)
        
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            raise

    def load_regular_model(self, model_size: str, model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """일반 모델 로딩"""
        start_time = time.time()
    
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=self.cache_dir,
            trust_remote_code=True  
        )
    
        # pad_token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        # 모델 로드 (H100에 최적화)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,  # H100에서 빠른 연산
            device_map="auto",          # 자동 GPU 배치
            trust_remote_code=True,
            max_memory={self.gpu_id: self.h100_config["max_memory_per_gpu"]} if self.gpu_id is not None else None
        )
    
        # 생성 설정
        model.generation_config = GenerationConfig(
            max_new_tokens=self.h100_config["max_new_tokens"],
            temperature=self.h100_config["temperature"],
            top_p=self.h100_config["top_p"],
            do_sample=self.h100_config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
        load_time = time.time() - start_time
    
        # 메모리 사용량 확인
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"✅ Model loaded in {load_time:.1f}s, GPU Memory: {memory_used:.1f}GB")
    
        # 캐시에 저장
        self.tokenizers[model_size] = tokenizer
        self.models[model_size] = model
    
        return tokenizer, model

    def load_kogpt_model(self, model_size: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """KoGPT 전용 로딩 함수"""
        start_time = time.time()
    
        # KoGPT 전용 토크나이저 설정
        tokenizer = AutoTokenizer.from_pretrained(
            'kakaobrain/kogpt',
            revision='KoGPT6B-ryan1.5b-float16',  # 중요: revision 지정
            bos_token='[BOS]',
            eos_token='[EOS]',
            unk_token='[UNK]',
            pad_token='[PAD]',
            mask_token='[MASK]',
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
    
        # KoGPT 전용 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            'kakaobrain/kogpt',
            revision='KoGPT6B-ryan1.5b-float16',  # 중요: revision 지정
            pad_token_id=tokenizer.eos_token_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
    
        # 생성 설정
        model.generation_config = GenerationConfig(
            max_new_tokens=self.h100_config["max_new_tokens"],
            temperature=self.h100_config["temperature"],
            top_p=self.h100_config["top_p"],
            do_sample=self.h100_config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
        load_time = time.time() - start_time
    
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"✅ KoGPT loaded in {load_time:.1f}s, GPU Memory: {memory_used:.1f}GB")
    
        # 캐시에 저장
        self.tokenizers[model_size] = tokenizer
        self.models[model_size] = model
    
        return tokenizer, model

    def load_kullm_model(self, model_size: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """KULLM 전용 로딩 함수 (오프로드 처리)"""
        start_time = time.time()
    
        # 오프로드 디렉토리 생성
        offload_dir = os.path.join(self.cache_dir, "offload")
        os.makedirs(offload_dir, exist_ok=True)
    
        # KULLM 토크나이저
        tokenizer = AutoTokenizer.from_pretrained(
            'nlpai-lab/kullm-polyglot-12.8b-v2',
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
    
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        # KULLM 모델 (오프로드 설정)
        model = AutoModelForCausalLM.from_pretrained(
            'nlpai-lab/kullm-polyglot-12.8b-v2',
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_dir,  # 중요: 오프로드 폴더 지정
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
    
        # 생성 설정
        model.generation_config = GenerationConfig(
            max_new_tokens=self.h100_config["max_new_tokens"],
            temperature=self.h100_config["temperature"],
            top_p=self.h100_config["top_p"],
            do_sample=self.h100_config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
        load_time = time.time() - start_time
    
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"✅ KULLM loaded in {load_time:.1f}s, GPU Memory: {memory_used:.1f}GB")
    
        # 캐시에 저장
        self.tokenizers[model_size] = tokenizer
        self.models[model_size] = model
    
        return tokenizer, model
    
    def generate_text(self, model_size: str, prompts: List[str]) -> List[Dict]:
        """텍스트 생성 및 성능 측정"""
        tokenizer, model = self.load_model(model_size)
        
        results = []
        batch_size = self.h100_config["batch_size"]
        
        logger.info(f"🚀 Generating with {model_size} model, batch_size={batch_size}")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # 토크나이징 (KULLM 호환성 개선)
            if model_size == "kullm":
                inputs = tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512,
                    return_token_type_ids=False  # KULLM은 token_type_ids 불필요
                ).to(self.device)
            else:
                inputs = tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                ).to(self.device)
            
            # 생성 시작
            start_time = time.time()
            torch.cuda.synchronize()  # 정확한 시간 측정
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.h100_config["max_new_tokens"],
                    temperature=self.h100_config["temperature"],
                    top_p=self.h100_config["top_p"],
                    do_sample=self.h100_config["do_sample"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            torch.cuda.synchronize()  # GPU 작업 완료 대기
            generation_time = time.time() - start_time
            
            # 디코딩
            generated_texts = []
            for j, output in enumerate(outputs):
                # 입력 제거하고 새로 생성된 부분만
                input_length = inputs['input_ids'][j].shape[0]
                generated = output[input_length:]
                generated_text = tokenizer.decode(generated, skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            # 결과 저장
            for j, (prompt, generated) in enumerate(zip(batch_prompts, generated_texts)):
                results.append({
                    "model_size": model_size,
                    "model_name": self.available_models[model_size],
                    "prompt": prompt,
                    "generated": generated,
                    "generation_time": generation_time / len(batch_prompts),  # 평균 시간
                    "tokens_generated": len(tokenizer.encode(generated)),
                    "tokens_per_second": len(tokenizer.encode(generated)) / (generation_time / len(batch_prompts)),
                    "timestamp": datetime.now().isoformat()
                })
        
        logger.info(f"✅ Generated {len(results)} responses with {model_size}")
        return results
    
    def run_korean_benchmark(self, model_sizes: List[str] = None):
        """한국어 벤치마크 실행"""
        if model_sizes is None:
            model_sizes = ["117M", "1.5B", "3.8B", "7B"]
        
        # 한국어 테스트 프롬프트들
        korean_prompts = [
            "한국의 전통 음식에 대해 설명해주세요.",
            "AI 기술의 미래에 대한 당신의 생각은 무엇인가요?",
            "기후 변화 문제를 해결하기 위한 방법을 제시해주세요.",
            "한국어 자연어 처리의 어려운 점은 무엇인가요?",
            "머신러닝과 딥러닝의 차이점을 설명해주세요.",
            "한국 문화가 세계에 미치는 영향에 대해 논해보세요.",
            "효율적인 학습 방법에 대한 조언을 해주세요.",
            "인공지능과 인간의 협력 관계는 어떻게 발전할까요?"
        ]
        
        logger.info(f"🇰🇷 Starting Korean benchmark with {len(korean_prompts)} prompts")
        logger.info(f"📊 Testing model sizes: {model_sizes}")
        
        all_results = []
        
        for model_size in model_sizes:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {model_size} model")
            logger.info(f"{'='*50}")
            
            try:
                results = self.generate_text(model_size, korean_prompts)
                all_results.extend(results)
                
                # 중간 결과 저장
                self.save_results(all_results, f"korean_benchmark_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                
            except Exception as e:
                logger.error(f"❌ Error testing {model_size}: {e}")
                continue
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 최종 결과 저장
        final_file = f"korean_benchmark_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_results(all_results, final_file)
        
        # 성능 분석
        self.analyze_results(all_results)
        
        return all_results
    
    def run_size_comparison(self):
        """M3 크기별 성능 비교 재현"""
        logger.info("🔬 M3 Size Comparison Reproduction on H100")
        
        # M3 원논문 스타일 프롬프트
        comparison_prompts = [
            "Translate the following English sentence to Korean: 'Artificial intelligence is transforming our world.'",
            "한국어로 답변해주세요: 인공지능의 장점과 단점을 비교해보세요.",
            "Solve this problem step by step: If a model has 1.5 billion parameters and another has 3.8 billion, which would you expect to perform better?",
            "한국의 수도는 어디인가요? 그리고 그 도시의 특징을 설명해주세요.",
            "Write a short story about a small model outperforming a larger model.",
            "다음 수학 문제를 해결하세요: 2^10 + 3^4 = ?",
            "Explain the concept of 'emergence' in language models.",
            "한국어 처리에서 토크나이제이션의 중요성에 대해 설명해주세요."
        ]
        
        # M3 핵심 비교: 1.5B vs 3.8B
        key_sizes = ["1.5B", "3.8B"]
        logger.info(f"🎯 Key M3 comparison: {key_sizes}")
        
        results = []
        for size in key_sizes:
            size_results = self.generate_text(size, comparison_prompts)
            results.extend(size_results)
        
        # 추가 크기들도 테스트
        additional_sizes = ["117M", "7B"]
        for size in additional_sizes:
            if size in self.available_models:
                size_results = self.generate_text(size, comparison_prompts)
                results.extend(size_results)
        
        # M3 스타일 분석
        self.analyze_m3_phenomenon(results)
        
        return results
    
    def analyze_results(self, results: List[Dict]):
        """결과 분석 및 시각화"""
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.warning("⚠️  No results to analyze")
            return
        
        logger.info("\n📊 Performance Analysis")
        logger.info("=" * 50)
        
        # 모델별 성능 요약
        summary = df.groupby(['model_size', 'model_name']).agg({
            'generation_time': ['mean', 'std'],
            'tokens_per_second': ['mean', 'std'],
            'tokens_generated': ['mean', 'std']
        }).round(3)
        
        print("\n📈 Performance Summary:")
        print(summary)
        
        # 가장 빠른 모델
        fastest_model = df.loc[df['tokens_per_second'].idxmax()]
        logger.info(f"\n🚀 Fastest model: {fastest_model['model_size']} ({fastest_model['tokens_per_second']:.1f} tokens/sec)")
        
        # 결과를 CSV로 저장
        csv_file = os.path.join(self.output_dir, f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary.to_csv(csv_file)
        logger.info(f"💾 Summary saved to: {csv_file}")
    
    def analyze_m3_phenomenon(self, results: List[Dict]):
        """M3 현상 분석"""
        df = pd.DataFrame(results)
        
        if df.empty:
            return
        
        logger.info("\n🔬 M3 Phenomenon Analysis")
        logger.info("=" * 50)
        
        # 1.5B vs 3.8B 직접 비교
        df_1_5b = df[df['model_size'] == '1.5B']
        df_3_8b = df[df['model_size'] == '3.8B']
        
        if not df_1_5b.empty and not df_3_8b.empty:
            avg_speed_1_5b = df_1_5b['tokens_per_second'].mean()
            avg_speed_3_8b = df_3_8b['tokens_per_second'].mean()
            
            logger.info(f"📊 1.5B average speed: {avg_speed_1_5b:.1f} tokens/sec")
            logger.info(f"📊 3.8B average speed: {avg_speed_3_8b:.1f} tokens/sec")
            
            if avg_speed_1_5b > avg_speed_3_8b:
                speed_advantage = ((avg_speed_1_5b - avg_speed_3_8b) / avg_speed_3_8b) * 100
                logger.info(f"🎯 M3 PHENOMENON CONFIRMED: 1.5B is {speed_advantage:.1f}% faster than 3.8B!")
            else:
                logger.info("🤔 No clear M3 phenomenon observed in speed")
        
        # 질적 분석을 위한 샘플 출력
        logger.info("\n📝 Sample outputs for quality comparison:")
        for size in ['1.5B', '3.8B']:
            if not df[df['model_size'] == size].empty:
                sample = df[df['model_size'] == size].iloc[0]
                logger.info(f"\n{size} Model Sample:")
                logger.info(f"Prompt: {sample['prompt'][:100]}...")
                logger.info(f"Generated: {sample['generated'][:200]}...")
    
    def save_results(self, results: List[Dict], filename: str):
        """결과를 JSON 파일로 저장"""
        filepath = os.path.join(self.output_dir, filename)
        
        # 메타데이터 추가
        output_data = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "total_gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A",
                "model_cache_dir": self.cache_dir,
                "h100_config": self.h100_config
            },
            "results": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Results saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="M3 Reproduction Experiment on H100")
    parser.add_argument("--experiment", choices=["korean", "m3", "both"], default="both",
                       help="Experiment type to run")
    parser.add_argument("--models", nargs="+", default=["117M", "1.5B", "3.8B", "7B"],
                       help="Model sizes to test")
    parser.add_argument("--cache-dir", default="./models", help="Model cache directory")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    
    args = parser.parse_args()
    
    # 실험 초기화
    experiment = M3ReproductionExperiment(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir
    )
    
    if args.batch_size:
        experiment.h100_config["batch_size"] = args.batch_size
    
    logger.info("🚀 Starting M3 Reproduction Experiment on H100")
    logger.info(f"📊 Models to test: {args.models}")
    logger.info(f"🎯 Experiment type: {args.experiment}")
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name()}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        if args.experiment in ["korean", "both"]:
            logger.info("\n🇰🇷 Starting Korean Benchmark...")
            experiment.run_korean_benchmark(args.models)
        
        if args.experiment in ["m3", "both"]:
            logger.info("\n🔬 Starting M3 Size Comparison...")
            experiment.run_size_comparison()
        
        logger.info("\n🎉 Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()