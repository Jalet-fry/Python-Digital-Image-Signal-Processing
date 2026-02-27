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
            
            # Логируем только внешний вызов (чтобы не тормозить на рекурсии)
            if DSPContext._call_depth == 1:
                # Путь: PythonDSP/results/debug_logs/var_X/labY/
                log_dir = os.path.join(BASE_DIR, "results", "debug_logs", f"var_{DSPContext.variant}", DSPContext.current_lab)
                
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                file_path = os.path.join(log_dir, f"{func.__name__}_output.txt")
                
                # Сохраняем результат
                if isinstance(result, np.ndarray) and result.size < 15000:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(f"--- {func.__name__.upper()} DUMP ---\n")
                        f.write(f"Variant: {DSPContext.variant}\n")
                        f.write(f"Shape: {result.shape}\n\n")
                        np.set_printoptions(threshold=np.inf, precision=6, suppress=True)
                        f.write(np.array2string(result))
                    
                    # Подтверждение в консоль (для спокойствия)
                    # print(f"  [LOG] Data saved to debug_logs/{func.__name__}_output.txt")
            
            return result
        finally:
            DSPContext._call_depth -= 1

    return wrapper
