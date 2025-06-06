# setup.py
"""
Packaging script for the *NeuroBulla Pipeline Bodies* project.
Key points vs. the original version
ᶠᶸᶜᵏᵧₒᵤ! Single-source version (read from ``nbpb/__init__.py``)  
⚚ Robust fallback if the version string is missing  
⚚ Explicit package discovery – include only the real library, exclude
   ``examples`` / ``tests``  
⚚ Requirements read from *requirements.txt* so you keep a single list
   of dependencies  
🌈⃤ Optional data-files support (examples) – keep it, but easy to drop
   if you prefer a leaner wheel
"""
from pathlib import Path
import re
from setuptools import setup, find_packages

PROJECT_ROOT = Path(__file__).parent
PACKAGE_NAME = "nbpb"

# Helpers
def read_version() -> str:
    """Extract __version__ from nbpb/__init__.py (single source of truth)."""
    version_file = PROJECT_ROOT / PACKAGE_NAME / "__init__.py"
    if version_file.exists():
        match = re.search(
            r'^__version__\s*=\s*["\']([^"\']+)["\']',
            version_file.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    # Fallback – don't silently publish "0.0.0" by mistake
    raise RuntimeError("Cannot find __version__ in nbpb/__init__.py")

def read_long_description() -> str:
    return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

def read_requirements() -> list[str]:
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        return [
            ln.strip()
            for ln in req_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    # Minimal hard-coded fallback – keeps `pip install .` working
    return [
        "torch>=1.8",
        "numpy>=1.19",
        "PyYAML>=5.0",
        "scikit-learn",
        "scipy",
    ]
# Setup
setup(
    name=PACKAGE_NAME,
    version=read_version(),
    author="Francesco Bulla",
    author_email="150francescobulla@gmail.com",
    description=(
        "NeuroBulla Pipeline Bodies – A biologically-inspired supervisory "
        "framework for machine-learning pipelines"
    ),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fra150/NeuroBulla-Pipeline-Bodies-NBPB-.git",
    python_requires=">=3.8",
    # Discover only the real library; keep wheels lean
    packages=find_packages(
        where=".",
        include=[f"{PACKAGE_NAME}", f"{PACKAGE_NAME}.*"],
        exclude=["examples", "examples.*", "tests", "tests.*"],
    ),

    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "nbpb=nbpb.cli:main",
        ],
    },

    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Optional: ship example YAML / scripts inside the wheel.
    # Delete this block if you prefer to keep the wheel minimal.
    include_package_data=True,
    package_data={
        "nbpb": ["examples/*.yaml", "examples/*.py"],
    },
)