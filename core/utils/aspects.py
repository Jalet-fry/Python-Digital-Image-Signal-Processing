import os
import numpy as np
import functools

# Определяем корень проекта для логов
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DSPContext:
    variant = 10
    current_lab = "lab1"
    _call_depth = 0

def log_dsp_action(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        DSPContext._call_depth += 1
        try:
            result = func(*args, **kwargs)
            
            # Логируем только вызовы верхнего уровня, чтобы не захламлять лог рекурсией
            if DSPContext._call_depth == 1:
                log_dir = os.path.join(BASE_DIR, "results", "debug_logs", f"var_{DSPContext.variant}", DSPContext.current_lab)
                
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                file_path = os.path.join(log_dir, f"{func.__name__}_output.txt")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"--- {func.__name__.upper()} DUMP ---\n")
                    f.write(f"Variant: {DSPContext.variant}\n")

                    def format_val(val):
                        if isinstance(val, np.ndarray):
                            np.set_printoptions(threshold=1000, precision=6, suppress=True)
                            return f"Array {val.shape}:\n{np.array2string(val)}"
                        return str(val)

                    if isinstance(result, tuple):
                        f.write(f"Returned {len(result)} values:\n")
                        for i, res in enumerate(result):
                            f.write(f"[{i}]: {format_val(res)}\n\n")
                    else:
                        f.write(f"Result: {format_val(result)}\n")
            
            return result
        finally:
            DSPContext._call_depth -= 1

    return wrapper
