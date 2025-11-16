"""
Script para limpar completamente o cache do Python.
Execute este script antes de treinar o modelo novamente.
"""

import os
import shutil
from pathlib import Path

def clean_pycache(root_dir='.'):
    """Remove todos os arquivos __pycache__ e .pyc"""
    root = Path(root_dir)
    removed_count = 0

    # Remover diretórios __pycache__
    for pycache_dir in root.rglob('__pycache__'):
        try:
            shutil.rmtree(pycache_dir)
            print(f"✓ Removido: {pycache_dir}")
            removed_count += 1
        except Exception as e:
            print(f"✗ Erro ao remover {pycache_dir}: {e}")

    # Remover arquivos .pyc individualmente
    for pyc_file in root.rglob('*.pyc'):
        try:
            pyc_file.unlink()
            print(f"✓ Removido: {pyc_file}")
            removed_count += 1
        except Exception as e:
            print(f"✗ Erro ao remover {pyc_file}: {e}")

    # Remover arquivos .pyo
    for pyo_file in root.rglob('*.pyo'):
        try:
            pyo_file.unlink()
            print(f"✓ Removido: {pyo_file}")
            removed_count += 1
        except Exception as e:
            print(f"✗ Erro ao remover {pyo_file}: {e}")

    print(f"\n{'='*60}")
    print(f"Limpeza concluída: {removed_count} itens removidos")
    print(f"{'='*60}")
    print("\nAgora você pode executar o treinamento novamente.")
    print("Exemplo: python scripts/train_model.py --symbol BTCUSDT --timeframe 15m")

if __name__ == "__main__":
    print("Limpando cache do Python...")
    print("="*60)
    clean_pycache()
