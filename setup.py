from setuptools import setup

with open("src/lemonade/version.py", encoding="utf-8") as fp:
    version = fp.read().split('"')[1]

setup(
    name="lemonade-sdk",
    version=version,
    description="Lemonade SDK: Your LLM Aide for Validation and Deployment",
    author_email="lemonade@amd.com",
    package_dir={"": "src"},
    packages=[
        "lemonade",
        "lemonade.profilers",
        "lemonade.common",
        "lemonade.tools",
        "lemonade.tools.huggingface",
        "lemonade.tools.oga",
        "lemonade.tools.llamacpp",
        "lemonade.tools.quark",
        "lemonade.tools.report",
        "lemonade.tools.server.utils",
        "lemonade.tools.server",
        "lemonade_install",
        "lemonade_server",
    ],
    install_requires=[
        # Minimal dependencies required for end-users who are running
        # apps deployed on Lemonade SDK
        "invoke>=2.0.0",
        "onnx>=1.11.0,<1.18.0",
        "pyyaml>=5.4",
        "typeguard>=2.3.13",
        "packaging>=20.9",
        # Necessary until upstream packages account for the breaking
        # change to numpy
        "numpy<2.0.0",
        "fasteners",
        "GitPython>=3.1.40",
        "psutil>=6.1.1",
        "wmi",
        "py-cpuinfo",
        "pytz",
        "zstandard",
        "fastapi",
        "uvicorn[standard]",
        "openai>=1.81.0",
        "transformers<=4.51.3",
        "jinja2",
        "tabulate",
        "sentencepiece",
        "huggingface-hub==0.33.0",
    ],
    extras_require={
        # The -minimal extras are meant to deploy specific backends into end-user
        # applications, without including developer-focused tools
        "oga-hybrid-minimal": [
            "onnxruntime-genai-directml-ryzenai==0.7.0.2",
            "protobuf>=6.30.1",
        ],
        "oga-cpu-minimal": [
            "onnxruntime-genai==0.8.2",
            "onnxruntime >=1.22.0",
        ],
        "llm": [
            # Minimal dependencies for developers to use all features of
            # Lemonade SDK, including building and optimizing models
            "torch>=2.6.0",
            "accelerate",
            "datasets",
            "pandas>=1.5.3",
            "matplotlib",
            # Install human-eval from a forked repo with Windows support until the
            # PR (https://github.com/openai/human-eval/pull/53) is merged
            "human-eval-windows==1.0.4",
            "lm-eval[api]",
        ],
        "llm-oga-cpu": [
            "lemonade-sdk[oga-cpu-minimal]",
            "lemonade-sdk[llm]",
        ],
        "llm-oga-igpu": [
            "onnxruntime-genai-directml==0.6.0",
            "onnxruntime-directml>=1.19.0,<1.22.0",
            "transformers<4.45.0",
            "lemonade-sdk[llm]",
        ],
        "llm-oga-cuda": [
            "onnxruntime-genai-cuda==0.8.2",
            "onnxruntime-gpu >=1.22.0",
            "transformers<=4.51.3",
            "lemonade-sdk[llm]",
        ],
        "llm-oga-npu": [
            "onnx==1.16.0",
            # NPU requires specific onnxruntime version for Ryzen AI compatibility
            # This may conflict with other OGA extras that require >=1.22.0
            "onnxruntime==1.18.0",
            "numpy==1.26.4",
            "protobuf>=6.30.1",
            "lemonade-sdk[llm]",
        ],
        "llm-oga-hybrid": [
            "lemonade-sdk[oga-hybrid-minimal]",
            "lemonade-sdk[llm]",
        ],
        "llm-oga-unified": [
            "lemonade-sdk[llm-oga-hybrid]",
        ],
    },
    classifiers=[],
    entry_points={
        "console_scripts": [
            "lemonade=lemonade:lemonadecli",
            "lemonade-install=lemonade_install:installcli",
            "lemonade-server-dev=lemonade_server.cli:main",
        ]
    },
    python_requires=">=3.10, <3.12",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "lemonade_server": ["server_models.json"],
        "lemonade": ["tools/server/static/*"],
    },
)

# This file was originally licensed under Apache 2.0. It has been modified.
# Modifications Copyright (c) 2025 AMD
