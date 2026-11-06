# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

chroma_datas = collect_data_files('chromadb')
pypdf_datas = collect_data_files('pypdf')

all_datas = chroma_datas + pypdf_datas

hidden_imports = [
    'core',
    'core.config',
    'core.database',
    'core.document_ops',
    'core.rag_engine',
    'core.hybrid_search',
    'core.citations',
    'core.memory',
    'core.history_db',
    'core.hardware',
    'core.health',
    'tui',
    'tui.menu',
    'sqlite3',
    'questionary',
    'rich',
    'prompt_toolkit',
    'tkinter',
    'tkinter.filedialog',
    'pypdf',
    'rank_bm25',
    'winreg',
    'ctypes',
] + collect_submodules('chromadb') + collect_submodules('langchain') + collect_submodules('langchain_community') + collect_submodules('langchain_core') + collect_submodules('langchain_ollama') + collect_submodules('langchain_classic') + collect_submodules('langchain_text_splitters') + collect_submodules('pypdf')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['fastapi', 'uvicorn', 'matplotlib', 'scipy', 'notebook', 'torch'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LocalAIHelper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

