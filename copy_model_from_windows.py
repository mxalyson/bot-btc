"""
Script para copiar modelo ML do ambiente Windows
Execute este script no Windows para copiar o modelo para o repositório
"""

import os
import shutil
from pathlib import Path

print("=" * 80)
print("📦 COPIANDO MODELO ML DO AMBIENTE WINDOWS")
print("=" * 80)
print()

# Possíveis localizações do modelo no Windows
possible_paths = [
    r"C:\Users\alyso\Downloads\bybit_scalping_bot\storage\models\ultra_scalper_btcusdt_365d.pkl",
    r"storage\models\ultra_scalper_btcusdt_365d.pkl",
    r"..\storage\models\ultra_scalper_btcusdt_365d.pkl",
    r"ultra_scalper_btcusdt_365d.pkl"
]

print("🔍 Procurando modelo ML...")
print()

model_found = None

for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Modelo encontrado em: {path}")
        model_found = path
        break
    else:
        print(f"❌ Não encontrado em: {path}")

print()

if not model_found:
    print("❌ ERRO: Modelo não encontrado em nenhuma das localizações!")
    print()
    print("Localizações verificadas:")
    for path in possible_paths:
        print(f"   - {path}")
    print()
    print("Soluções:")
    print("   1. Certifique-se de que o modelo foi treinado")
    print("   2. Execute um dos scripts de validação:")
    print("      python validate_ultra_optimized_FINAL.py")
    print("   3. O modelo será salvo automaticamente")
    print()
    input("Pressione ENTER para sair...")
    exit(1)

# Destino
destination = Path("storage/models/ultra_scalper_btcusdt_365d.pkl")

# Criar diretório se não existir
destination.parent.mkdir(parents=True, exist_ok=True)

print(f"📋 Copiando modelo...")
print(f"   De: {model_found}")
print(f"   Para: {destination}")
print()

try:
    shutil.copy2(model_found, destination)
    print("✅ Modelo copiado com sucesso!")
    print()

    # Verificar tamanho
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"📊 Tamanho do modelo: {size_mb:.2f} MB")

    if size_mb < 1:
        print()
        print("⚠️  AVISO: Modelo parece muito pequeno (< 1 MB)")
        print("   Verifique se o arquivo está correto.")
    elif size_mb > 500:
        print()
        print("⚠️  AVISO: Modelo muito grande (> 500 MB)")
        print("   Isso pode indicar um problema.")
    else:
        print("✅ Tamanho do modelo OK")

    print()
    print("🚀 Próximos passos:")
    print("   1. Execute: python setup.py")
    print("   2. Configure o .env")
    print("   3. Execute: python main.py")
    print()

except Exception as e:
    print(f"❌ ERRO ao copiar modelo: {e}")
    print()
    input("Pressione ENTER para sair...")
    exit(1)

print("=" * 80)
print()
input("Pressione ENTER para sair...")
