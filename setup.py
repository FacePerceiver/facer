from os import path as os_path

from setuptools import setup, find_packages

this_directory = os_path.abspath(os_path.dirname(__file__))


# 读取文件内容
def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding="utf-8") as f:
        long_description = f.read()
    return long_description


# 获取依赖
def read_requirements(filename):
    return [
        line.strip()
        for line in read_file(filename).splitlines()
        if not line.startswith("#")
    ]

def get_version():
    version_file = 'facer/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

setup(
    name="facer",
    version=get_version(),
    description="Face related toolkit",
    author="FacePerceiver",
    author_email="admin@hypercube.top",
    url="https://github.com/FacePerceiver/facer",
    license="MIT",
    keywords="face-detection pytorch RetinaFace face-parsing farl",
    project_urls={
        "Documentation": "https://github.com/FacePerceiver/facer",
        "Source": "https://github.com/FacePerceiver/facer",
        "Tracker": "https://github.com/FacePerceiver/facer/issues",
    },
    long_description=read_file("README.md"),  # 读取的Readme文档内容
    long_description_content_type="text/markdown",  # 指定包文档格式为markdown
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["numpy", "torch", "torchvision", "opencv-python","validators"]
)