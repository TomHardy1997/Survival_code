import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

class H5FileChecker:
    """H5文件完整性检查工具"""
    
    def __init__(self, h5_dir, output_dir='./h5_check_results'):
        """
        Args:
            h5_dir: H5文件目录
            output_dir: 检查结果保存目录
        """
        self.h5_dir = h5_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集所有H5文件
        self.h5_files = self._collect_h5_files()
        print(f"Found {len(self.h5_files)} H5 files in {h5_dir}")
    
    def _collect_h5_files(self):
        """收集所有H5文件"""
        h5_files = []
        for root, dirs, files in os.walk(self.h5_dir):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
        return sorted(h5_files)
    
    def check_single_file(self, h5_path):
        """
        检查单个H5文件
        
        Returns:
            dict: 检查结果
        """
        result = {
            'file_path': h5_path,
            'file_name': os.path.basename(h5_path),
            'status': 'unknown',
            'error': None,
            'file_size_mb': 0,
            'has_features': False,
            'has_coords': False,
            'features_shape': None,
            'coords_shape': None,
            'features_dtype': None,
            'coords_dtype': None,
            'num_patches': 0,
            'shape_match': False,
            'is_empty': False,
            'has_nan': False,
            'has_inf': False,
            'features_min': None,
            'features_max': None,
            'features_mean': None,
        }
        
        try:
            # 检查文件是否存在
            if not os.path.exists(h5_path):
                result['status'] = 'not_found'
                result['error'] = 'File not found'
                return result
            
            # 检查文件大小
            file_size = os.path.getsize(h5_path)
            result['file_size_mb'] = file_size / (1024 * 1024)
            
            if file_size == 0:
                result['status'] = 'empty_file'
                result['error'] = 'File size is 0'
                return result
            
            # 尝试打开文件
            try:
                with h5py.File(h5_path, 'r') as f:
                    # 检查必需的keys
                    result['has_features'] = 'features' in f
                    result['has_coords'] = 'coords' in f
                    
                    if not result['has_features']:
                        result['status'] = 'missing_features'
                        result['error'] = 'Missing "features" dataset'
                        return result
                    
                    if not result['has_coords']:
                        result['status'] = 'missing_coords'
                        result['error'] = 'Missing "coords" dataset'
                        return result
                    
                    # 读取数据
                    features = f['features'][:]
                    coords = f['coords'][:]
                    
                    # 记录形状和类型
                    result['features_shape'] = features.shape
                    result['coords_shape'] = coords.shape
                    result['features_dtype'] = str(features.dtype)
                    result['coords_dtype'] = str(coords.dtype)
                    
                    # 检查是否为空
                    if features.shape[0] == 0:
                        result['status'] = 'empty_data'
                        result['error'] = 'Features array is empty'
                        result['is_empty'] = True
                        return result
                    
                    result['num_patches'] = features.shape[0]
                    
                    # 检查形状是否匹配
                    result['shape_match'] = (features.shape[0] == coords.shape[0])
                    if not result['shape_match']:
                        result['status'] = 'shape_mismatch'
                        result['error'] = f'Shape mismatch: features {features.shape[0]} vs coords {coords.shape[0]}'
                        return result
                    
                    # 检查NaN和Inf
                    result['has_nan'] = bool(np.isnan(features).any())
                    result['has_inf'] = bool(np.isinf(features).any())
                    
                    if result['has_nan']:
                        result['status'] = 'has_nan'
                        result['error'] = 'Features contain NaN values'
                        return result
                    
                    if result['has_inf']:
                        result['status'] = 'has_inf'
                        result['error'] = 'Features contain Inf values'
                        return result
                    
                    # 统计信息
                    result['features_min'] = float(np.min(features))
                    result['features_max'] = float(np.max(features))
                    result['features_mean'] = float(np.mean(features))
                    
                    # 一切正常
                    result['status'] = 'valid'
                    
            except OSError as e:
                result['status'] = 'corrupted'
                result['error'] = f'OSError: {str(e)}'
                return result
            
            except Exception as e:
                result['status'] = 'read_error'
                result['error'] = f'{type(e).__name__}: {str(e)}'
                return result
        
        except Exception as e:
            result['status'] = 'unknown_error'
            result['error'] = f'Unexpected error: {str(e)}'
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def check_all_files(self, num_workers=8, save_interval=100):
        """
        检查所有H5文件
        
        Args:
            num_workers: 并行进程数
            save_interval: 每检查多少个文件保存一次结果
        """
        print(f"\nChecking {len(self.h5_files)} H5 files...")
        print(f"Using {num_workers} workers")
        
        results = []
        
        # 使用进程池并行检查
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.check_single_file, h5_path): h5_path 
                for h5_path in self.h5_files
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(self.h5_files), desc="Checking files") as pbar:
                for i, future in enumerate(as_completed(future_to_file)):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        h5_path = future_to_file[future]
                        results.append({
                            'file_path': h5_path,
                            'file_name': os.path.basename(h5_path),
                            'status': 'check_failed',
                            'error': str(e)
                        })
                    
                    pbar.update(1)
                    
                    # 定期保存结果
                    if (i + 1) % save_interval == 0:
                        self._save_intermediate_results(results, i + 1)
        
        # 保存最终结果
        self._save_final_results(results)
        
        return results
    
    def _save_intermediate_results(self, results, count):
        """保存中间结果"""
        df = pd.DataFrame(results)
        output_file = os.path.join(self.output_dir, f'intermediate_results_{count}.csv')
        df.to_csv(output_file, index=False)
    
    def _save_final_results(self, results):
        """保存最终结果并生成报告"""
        df = pd.DataFrame(results)
        
        # 保存完整结果
        full_output = os.path.join(self.output_dir, 'h5_check_full_results.csv')
        df.to_csv(full_output, index=False)
        print(f"\n✅ Full results saved to: {full_output}")
        
        # 统计各种状态
        status_counts = df['status'].value_counts()
        
        # 生成摘要报告
        report = []
        report.append("=" * 80)
        report.append("H5 Files Check Summary")
        report.append("=" * 80)
        report.append(f"Total files checked: {len(results)}")
        report.append(f"\nStatus breakdown:")
        for status, count in status_counts.items():
            percentage = count / len(results) * 100
            report.append(f"  {status:20s}: {count:6d} ({percentage:5.2f}%)")
        
        # 有效文件统计
        valid_df = df[df['status'] == 'valid']
        if len(valid_df) > 0:
            report.append(f"\n✅ Valid files: {len(valid_df)}")
            report.append(f"  Total patches: {valid_df['num_patches'].sum():,}")
            report.append(f"  Avg patches per file: {valid_df['num_patches'].mean():.1f}")
            report.append(f"  Min patches: {valid_df['num_patches'].min()}")
            report.append(f"  Max patches: {valid_df['num_patches'].max()}")
            report.append(f"  Total size: {valid_df['file_size_mb'].sum():.2f} MB")
        
        # 问题文件统计
        problem_df = df[df['status'] != 'valid']
        if len(problem_df) > 0:
            report.append(f"\n⚠️  Problem files: {len(problem_df)}")
            
            # 保存问题文件列表
            problem_output = os.path.join(self.output_dir, 'problem_files.csv')
            problem_df.to_csv(problem_output, index=False)
            report.append(f"  Details saved to: {problem_output}")
            
            # 按错误类型分组
            report.append(f"\n  Problem breakdown:")
            for status in problem_df['status'].unique():
                count = len(problem_df[problem_df['status'] == status])
                report.append(f"    {status}: {count}")
        
        report.append("=" * 80)
        
        # 打印报告
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        # 保存报告
        report_file = os.path.join(self.output_dir, 'check_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"\n📄 Report saved to: {report_file}")
        
        return df, problem_df
    
    def quick_check(self, sample_size=100):
        """
        快速检查（随机抽样）
        
        Args:
            sample_size: 抽样数量
        """
        import random
        
        sample_files = random.sample(self.h5_files, min(sample_size, len(self.h5_files)))
        
        print(f"\nQuick check: sampling {len(sample_files)} files...")
        
        results = []
        for h5_path in tqdm(sample_files, desc="Checking"):
            result = self.check_single_file(h5_path)
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # 打印快速统计
        print("\n" + "=" * 60)
        print("Quick Check Results")
        print("=" * 60)
        print(f"Sample size: {len(sample_files)}")
        print("\nStatus breakdown:")
        print(df['status'].value_counts())
        
        valid_count = len(df[df['status'] == 'valid'])
        print(f"\n✅ Valid: {valid_count}/{len(sample_files)} ({valid_count/len(sample_files)*100:.1f}%)")
        
        if valid_count < len(sample_files):
            print("\n⚠️  Found problems! Run full check for details.")
        
        return df


def check_specific_files(file_list, output_dir='./h5_check_results'):
    """
    检查指定的H5文件列表
    
    Args:
        file_list: H5文件路径列表
        output_dir: 结果保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checker = H5FileChecker.__new__(H5FileChecker)
    checker.output_dir = output_dir
    
    results = []
    for h5_path in tqdm(file_list, desc="Checking files"):
        result = checker.check_single_file(h5_path)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # 保存结果
    output_file = os.path.join(output_dir, 'specific_files_check.csv')
    df.to_csv(output_file, index=False)
    
    # 打印统计
    print("\n" + "=" * 60)
    print("Check Results")
    print("=" * 60)
    print(f"Total files: {len(file_list)}")
    print("\nStatus breakdown:")
    print(df['status'].value_counts())
    print(f"\nResults saved to: {output_file}")
    
    return df


# ============= 使用示例 =============
if __name__ == '__main__':
    # 方式1: 检查整个目录
    checker = H5FileChecker(
        h5_dir='/home/stat-jijianxin/PFMs/HMU_GC_ALL_H5/features_ctranspath',
        output_dir='./h5_check_results'
    )
    
    # 选项A: 快速检查（抽样100个文件）
    print("\n🔍 Running quick check...")
    quick_results = checker.quick_check(sample_size=100)
    
    # 如果快速检查发现问题，再运行完整检查
    if len(quick_results[quick_results['status'] != 'valid']) > 0:
        print("\n⚠️  Quick check found problems. Running full check...")
        user_input = input("Continue with full check? (y/n): ")
        if user_input.lower() == 'y':
            full_results, problem_files = checker.check_all_files(num_workers=8)
    else:
        print("\n✅ Quick check passed! All sampled files are valid.")
        user_input = input("Run full check anyway? (y/n): ")
        if user_input.lower() == 'y':
            full_results, problem_files = checker.check_all_files(num_workers=8)
    
    # 方式2: 只检查特定文件
    # specific_files = [
    #     '/path/to/file1.h5',
    #     '/path/to/file2.h5',
    # ]
    # results = check_specific_files(specific_files)
    
    # 方式3: 检查CSV中列出的文件
    # import pandas as pd
    # csv_df = pd.read_csv('your_csv_file.csv')
    # slide_ids = csv_df['slide_id'].tolist()
    # h5_paths = [os.path.join(h5_dir, f"{sid}.h5") for sid in slide_ids]
    # results = check_specific_files(h5_paths)
