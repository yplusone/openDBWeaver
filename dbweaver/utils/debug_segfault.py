# coding: utf-8
"""
debug_segfault.py
-----------------
调试工具：帮助定位 segmentation fault 的具体位置

提供多种调试方法：
1. 使用 gdb 获取堆栈跟踪
2. 使用 valgrind 检查内存错误
3. 使用 AddressSanitizer (ASan) 编译和运行
4. 生成 core dump 并分析
"""

import subprocess
import tempfile
import os
import shlex
from pathlib import Path
from typing import Optional, Tuple
from config import DUCKDB_BINARY_PATH, DB_PATH


class SegfaultDebugger:
    """调试 segmentation fault 的工具类"""
    
    def __init__(self, duckdb_binary: str = DUCKDB_BINARY_PATH + "debug/duckdb", db_path: str = DB_PATH):
        self.duckdb_binary = duckdb_binary
        self.db_path = db_path
    
    def debug_with_gdb(self, query: str, pragmas: list[str] = None) -> Tuple[bool, str]:
        """
        使用 gdb 运行查询并获取堆栈跟踪
        
        Returns:
            (success: bool, stack_trace: str)
        """
        if pragmas is None:
            pragmas = ["PRAGMA threads = 1;"]
        
        pragma_sql = " ".join(
            p.strip() if p.strip().endswith(";") else f"{p.strip()};"
            for p in pragmas
        )
        combined_sql = f"{pragma_sql} {query}"
        
        # 将 SQL 写入临时文件，然后通过文件路径传递，避免引号转义问题
        sql_file = None
        try:
            # 将 SQL 写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False, encoding='utf-8') as sql_f:
                sql_f.write(combined_sql)
                sql_file = sql_f.name
            
            # 使用 GDB 的 Python API 或者通过文件读取 SQL
            # 方法：使用 GDB 的 -ex 选项，通过 shell 命令读取文件
            # 但更简单的方法：直接使用 subprocess 的 shell=True，让 shell 处理引号
            # 或者：使用环境变量传递 SQL
            
            # 读取 SQL 文件内容
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read().strip()
            
            # 构建 GDB 命令脚本
            # 不在脚本中设置 args，使用 --args 选项传递参数
            gdb_script_content = """set confirm off
set pagination off
run
bt
info registers
quit
"""
            
            # 写入 GDB 脚本文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False, encoding='utf-8') as gdb_f:
                gdb_f.write(gdb_script_content)
                gdb_script = gdb_f.name
            
            # 构建 GDB 命令
            # 使用 --args 选项直接传递参数，让 GDB 自己处理参数解析
            # 这样 SQL 内容不会被当作 GDB 命令执行
            cmd = [
                "gdb",
                "-batch",
                "-x", gdb_script,
                "--args",  # 使用 --args 选项
                self.duckdb_binary,
                self.db_path,
                "-c",
                sql_content,  # 直接传递 SQL 内容，subprocess 会处理转义
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60秒超时
            )
            
            # 提取堆栈跟踪
            output = result.stdout + result.stderr
            
            # 查找堆栈跟踪部分
            if "Program received signal SIGSEGV" in output or "Segmentation fault" in output:
                # 提取从 "Program received signal" 到结束的内容
                lines = output.split('\n')
                start_idx = None
                for i, line in enumerate(lines):
                    if "Program received signal" in line or "SIGSEGV" in line:
                        start_idx = i
                        break
                
                if start_idx is not None:
                    stack_trace = '\n'.join(lines[start_idx:])
                    return True, stack_trace
            
            return False, output
            
        except subprocess.TimeoutExpired:
            return False, "GDB timeout after 60 seconds"
        except FileNotFoundError:
            return False, "gdb not found. Please install gdb: sudo apt-get install gdb"
        finally:
            # 清理临时文件
            if sql_file and os.path.exists(sql_file):
                try:
                    os.unlink(sql_file)
                except:
                    pass
            if 'gdb_script' in locals() and gdb_script and os.path.exists(gdb_script):
                try:
                    os.unlink(gdb_script)
                except:
                    pass
    
    def debug_with_valgrind(self, query: str, pragmas: list[str] = None) -> Tuple[bool, str]:
        """
        使用 valgrind 检查内存错误
        
        Returns:
            (success: bool, valgrind_report: str)
        """
        if pragmas is None:
            pragmas = ["PRAGMA threads = 1;"]
        
        pragma_sql = " ".join(
            p.strip() if p.strip().endswith(";") else f"{p.strip()};"
            for p in pragmas
        )
        combined_sql = f"{pragma_sql} {query}"
        
        cmd = [
            "valgrind",
            "--leak-check=full",
            "--show-leak-kinds=all",
            "--track-origins=yes",
            "--verbose",
            "--error-limit=no",
            self.duckdb_binary,
            self.db_path,
            "-c", combined_sql,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2分钟超时
            )
            
            # valgrind 的输出在 stderr
            output = result.stderr
            
            # 查找错误报告
            if "Invalid read" in output or "Invalid write" in output or "Use of uninitialised value" in output:
                # 提取错误部分
                lines = output.split('\n')
                error_lines = []
                in_error = False
                for line in lines:
                    if "Invalid read" in line or "Invalid write" in line or "Use of uninitialised value" in line:
                        in_error = True
                        error_lines.append(line)
                    elif in_error and line.strip() and not line.startswith("=="):
                        error_lines.append(line)
                    elif in_error and line.startswith("==") and "ERROR SUMMARY" in line:
                        error_lines.append(line)
                        break
                
                return True, '\n'.join(error_lines)
            
            return False, output
            
        except subprocess.TimeoutExpired:
            return False, "Valgrind timeout after 120 seconds"
        except FileNotFoundError:
            return False, "valgrind not found. Please install valgrind: sudo apt-get install valgrind"
    
    def _check_debug_symbols(self, binary_path: str) -> bool:
        """检查二进制文件是否包含调试符号"""
        try:
            # 使用 readelf 检查是否有调试符号
            cmd = ["readelf", "-S", binary_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # 检查是否有 .debug_info 或 .debug_line 段
                return ".debug_info" in result.stdout or ".debug_line" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False
    
    def get_backtrace_with_addr2line(self, addresses: list[str], binary_path: str = None) -> str:
        """
        使用 addr2line 将地址转换为文件名和行号
        
        Args:
            addresses: 地址列表（从堆栈跟踪中提取）
            binary_path: 二进制文件路径（默认使用 duckdb_binary）
        
        Returns:
            格式化的源代码位置信息
        """
        if binary_path is None:
            binary_path = self.duckdb_binary
        
        if not os.path.exists(binary_path):
            return f"Binary not found: {binary_path}"
        
        # 检查是否有调试符号
        has_debug_symbols = self._check_debug_symbols(binary_path)
        if not has_debug_symbols:
            return (
                f"Warning: Binary {binary_path} appears to have no debug symbols.\n"
                f"To get source line information, rebuild with debug symbols:\n"
                f"  cmake -DCMAKE_BUILD_TYPE=Debug ...\n"
                f"  or add -g flag to compiler\n\n"
                f"Addresses: {', '.join(addresses[:5])}..."
            )
        
        results = []
        unknown_count = 0
        for addr in addresses:
            try:
                # 确保地址格式正确（添加 0x 前缀如果缺失）
                if not addr.startswith('0x'):
                    addr = '0x' + addr
                
                cmd = ["addr2line", "-e", binary_path, "-f", "-C", addr]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        func = lines[0]
                        location = lines[1]
                        
                        # 检查是否是未知地址（问号表示）
                        if func == "??" or location == "??:0" or location == "??:?":
                            unknown_count += 1
                            # 尝试使用 objdump 获取符号信息
                            symbol_info = self._get_symbol_info(binary_path, addr)
                            if symbol_info:
                                results.append(f"{addr}: {symbol_info}")
                            else:
                                results.append(
                                    f"{addr}: ?? (address not in binary or in dynamic library)\n"
                                    f"  Note: This address might be in a shared library. "
                                    f"Use 'info sharedlibrary' in gdb to find the library."
                                )
                        else:
                            results.append(f"{addr}: {func} at {location}")
                else:
                    results.append(f"{addr}: addr2line failed - {result.stderr.strip()}")
            except FileNotFoundError:
                return "addr2line not found. Please install binutils: sudo apt-get install binutils"
            except subprocess.TimeoutExpired:
                results.append(f"{addr}: addr2line timeout")
            except Exception as e:
                results.append(f"{addr}: Error - {e}")
        
        if unknown_count > 0 and len(addresses) > 0:
            results.insert(0, 
                f"Note: {unknown_count}/{len(addresses)} addresses could not be resolved.\n"
                f"This usually means:\n"
                f"  1. Addresses are in shared libraries (libc, libstdc++, etc.)\n"
                f"  2. Binary was stripped or compiled without debug symbols\n"
                f"  3. Addresses are outside the binary's address space\n"
                f"Use 'info sharedlibrary' in gdb to identify which library contains the address.\n"
            )
        
        return '\n'.join(results)
    
    def _get_symbol_info(self, binary_path: str, addr: str) -> Optional[str]:
        """使用 objdump 尝试获取符号信息（即使没有调试符号）"""
        try:
            # 移除 0x 前缀用于 objdump
            addr_clean = addr.replace('0x', '')
            cmd = ["objdump", "-T", binary_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # 查找最接近的符号
                # 这里简化处理，实际需要解析地址范围
                lines = result.stdout.split('\n')
                for line in lines:
                    if addr_clean.lower() in line.lower():
                        return f"Symbol: {line.strip()}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None
    
    def extract_addresses_from_stacktrace(self, stacktrace: str) -> list[str]:
        """
        从堆栈跟踪中提取地址
        
        Args:
            stacktrace: gdb 堆栈跟踪输出
        
        Returns:
            地址列表
        """
        import re
        # 匹配类似 "#0  0x00007ffff7a12345 in function" 的格式
        pattern = r'#\d+\s+0x([0-9a-fA-F]+)'
        addresses = re.findall(pattern, stacktrace)
        return addresses
    
    def comprehensive_debug(self, query: str, pragmas: list[str] = None) -> str:
        """
        综合调试：尝试多种方法并返回详细报告
        
        Returns:
            完整的调试报告
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SEGFAULT DEBUG REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nQuery: {query}\n")
        
        # 1. 尝试 gdb
        report_lines.append("\n[1] GDB Stack Trace:")
        report_lines.append("-" * 80)
        success, gdb_output = self.debug_with_gdb(query, pragmas)
        if success:
            report_lines.append(gdb_output)
            
            # 提取地址并转换
            addresses = self.extract_addresses_from_stacktrace(gdb_output)
            if addresses:
                report_lines.append("\n[1.1] Source Code Locations:")
                report_lines.append("-" * 80)
                locations = self.get_backtrace_with_addr2line(addresses[:10])  # 只转换前10个
                report_lines.append(locations)
        else:
            report_lines.append(f"GDB failed: {gdb_output}")
        
        # 2. 尝试 valgrind
        report_lines.append("\n\n[2] Valgrind Memory Check:")
        report_lines.append("-" * 80)
        success, valgrind_output = self.debug_with_valgrind(query, pragmas)
        if success:
            report_lines.append(valgrind_output)
        else:
            report_lines.append(f"Valgrind failed: {valgrind_output}")
        
        report_lines.append("\n" + "=" * 80)
        
        return '\n'.join(report_lines)


def debug_segfault(query: str, duckdb_binary: str = None, db_path: str = None) -> str:
    """
    便捷函数：调试 segmentation fault
    
    Usage:
        report = debug_segfault("SELECT * FROM dbweaver(...);")
        print(report)
    """
    debugger = SegfaultDebugger(
        duckdb_binary=duckdb_binary or DUCKDB_BINARY_PATH + "debug/duckdb",
        db_path=db_path or DB_PATH
    )
    return debugger.comprehensive_debug(query)


if __name__ == "__main__":
    # 示例用法
    test_query = "SELECT * FROM dbweaver((SELECT l_suppkey FROM lineitem LIMIT 10));"
    report = debug_segfault(test_query)
    print(report)

