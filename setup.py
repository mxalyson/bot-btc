"""
Setup Script - Valida e prepara ambiente para mainnet
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("🚀 BTC SCALPER BOT - SETUP MAINNET")
print("=" * 80)
print()

# Cores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check(condition, message):
    """Check e print resultado."""
    if condition:
        print(f"{GREEN}✅ {message}{RESET}")
        return True
    else:
        print(f"{RED}❌ {message}{RESET}")
        return False

def warn(message):
    """Print warning."""
    print(f"{YELLOW}⚠️  {message}{RESET}")

errors = []
warnings = []

print("📋 VALIDANDO AMBIENTE...")
print()

# 1. Check Python version
print("1️⃣  Python Version")
py_version = sys.version_info
if check(py_version >= (3, 8), f"Python {py_version.major}.{py_version.minor}.{py_version.micro}"):
    pass
else:
    errors.append("Python 3.8+ necessário")

print()

# 2. Check diretórios
print("2️⃣  Estrutura de Diretórios")
directories = [
    'core',
    'storage',
    'storage/models',
    'logs'
]

for directory in directories:
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        warn(f"Diretório '{directory}' criado")
    check(path.exists(), f"Diretório '{directory}' existe")

print()

# 3. Check arquivos core
print("3️⃣  Arquivos Core")
core_files = [
    'main.py',
    'core/__init__.py',
    'core/bybit_api.py',
    'core/trading_bot.py',
    'core/telegram_bot.py',
    'config_ultra_optimized_FINAL.yaml',
    'requirements.txt'
]

for file in core_files:
    if not check(Path(file).exists(), f"Arquivo '{file}'"):
        errors.append(f"Arquivo '{file}' não encontrado")

print()

# 4. Check .env
print("4️⃣  Configuração (.env)")
env_exists = Path('.env').exists()
env_example_exists = Path('.env.example').exists()

if not env_exists:
    if env_example_exists:
        warn("Arquivo .env NÃO existe! Copie de .env.example")
        print()
        print("   Execute: cp .env.example .env")
        print("   Depois edite .env com suas credenciais")
        print()
        errors.append("Arquivo .env não encontrado")
    else:
        errors.append("Arquivo .env.example não encontrado")
else:
    check(True, "Arquivo .env existe")

    # Validar campos do .env
    with open('.env', 'r') as f:
        env_content = f.read()

    required_fields = [
        'TRADING_MODE',
        'BYBIT_API_KEY',
        'BYBIT_API_SECRET',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]

    for field in required_fields:
        if field in env_content:
            value = [line for line in env_content.split('\n') if line.startswith(field)]
            if value:
                val = value[0].split('=')[1].strip()
                if val and val not in ['your_api_key_here', 'your_api_secret_here', 'your_telegram_bot_token', 'your_telegram_user_id_here']:
                    check(True, f"{field} configurado")
                else:
                    warn(f"{field} ainda está com valor padrão")
                    warnings.append(f"Configure {field} no arquivo .env")
        else:
            warn(f"{field} não encontrado no .env")
            warnings.append(f"Adicione {field} ao arquivo .env")

print()

# 5. Check modelo ML
print("5️⃣  Modelo ML")
model_path = Path('storage/models/ultra_scalper_btcusdt_365d.pkl')

if check(model_path.exists(), f"Modelo ML '{model_path}'"):
    # Check tamanho
    size_mb = model_path.stat().st_size / (1024 * 1024)
    if size_mb > 1:
        check(True, f"Modelo tem {size_mb:.1f} MB (tamanho OK)")
    else:
        warn(f"Modelo tem apenas {size_mb:.1f} MB (pode estar corrompido)")
        warnings.append("Modelo muito pequeno, verifique se está correto")
else:
    errors.append("Modelo ML não encontrado")
    print()
    print(f"{RED}   ERRO CRÍTICO: Modelo ML não encontrado!{RESET}")
    print()
    print("   O bot NÃO vai funcionar sem o modelo.")
    print()
    print("   Soluções:")
    print("   1. Copie o modelo do seu ambiente Windows:")
    print("      C:\\Users\\alyso\\Downloads\\bybit_scalping_bot\\storage\\models\\ultra_scalper_btcusdt_365d.pkl")
    print()
    print("   2. Ou treine um novo modelo executando:")
    print("      python validate_ultra_optimized_FINAL.py")
    print()

print()

# 6. Check dependências Python
print("6️⃣  Dependências Python")

required_packages = [
    'pandas',
    'numpy',
    'yaml',
    'requests',
    'dotenv',
    'telegram',
    'sklearn',
    'lightgbm',
    'xgboost'
]

missing_packages = []

for package in required_packages:
    try:
        if package == 'yaml':
            __import__('yaml')
        elif package == 'dotenv':
            __import__('dotenv')
        elif package == 'telegram':
            __import__('telegram')
        elif package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        check(True, f"Pacote '{package}' instalado")
    except ImportError:
        warn(f"Pacote '{package}' NÃO instalado")
        missing_packages.append(package)

if missing_packages:
    print()
    print(f"{YELLOW}   Instale as dependências faltantes:{RESET}")
    print()
    print("   pip install -r requirements.txt")
    print()
    warnings.append("Dependências Python faltando")

print()

# 7. Resumo
print("=" * 80)
print("📊 RESUMO")
print("=" * 80)
print()

if errors:
    print(f"{RED}❌ ERROS CRÍTICOS ({len(errors)}):{RESET}")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    print()

if warnings:
    print(f"{YELLOW}⚠️  AVISOS ({len(warnings)}):{RESET}")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")
    print()

if not errors and not warnings:
    print(f"{GREEN}✅ TUDO OK! Ambiente pronto para mainnet!{RESET}")
    print()
    print("🚀 Próximos passos:")
    print()
    print("   1. Revise o arquivo .env (principalmente TRADING_MODE)")
    print("   2. Execute: python main.py")
    print("   3. Use /start no Telegram para controlar o bot")
    print()
    print("⚠️  IMPORTANTE: Sempre comece em PAPER TRADING MODE!")
    print()
elif not errors:
    print(f"{YELLOW}⚠️  Ambiente quase pronto, mas há avisos.{RESET}")
    print()
    print("   Corrija os avisos acima e execute setup.py novamente.")
    print()
else:
    print(f"{RED}❌ Ambiente NÃO está pronto!{RESET}")
    print()
    print("   Corrija os erros acima antes de executar o bot.")
    print()
    sys.exit(1)

print("=" * 80)
