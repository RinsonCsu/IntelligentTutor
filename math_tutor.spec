# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect sympy and torch data files
datas = []
datas += collect_data_files('sympy')
datas += collect_data_files('torch')

# Bundle the trained error classifier weights
datas += [('error_classifier.pt', '.')]

# Bundle the word problem model if present
wp_model_dir = 'word_problem_model'
if os.path.isdir(wp_model_dir):
    datas += [(wp_model_dir, 'word_problem_model')]

hidden_imports = []
hidden_imports += collect_submodules('sympy')
hidden_imports += collect_submodules('torch')
hidden_imports += [
    'sympy.parsing.sympy_parser',
    'sympy.parsing.sympy_tokenize',
    'sympy.core',
    'sympy.solvers',
    'tkinter',
    'tkinter.font',
    'tkinter.ttk',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pandas', 'PIL', 'cv2', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MathTutor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MathTutor',
)

app = BUNDLE(
    coll,
    name='MathTutor.app',
    icon=None,
    bundle_identifier='com.mathtutor.app',
    info_plist={
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleName': 'MathTutor',
    },
)
