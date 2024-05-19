from setuptools import setup, find_packages

setup(
    name="policy_eval",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.5",
        "gym==0.23.0",
        "h5py==3.1.0",
        "mujoco-py==2.1.2.14",
        "pandas==2.0.3",
        "tqdm==4.66.1",
        "click==8.1.7",
        "termcolor==1.1.0",
        "tensorflow-addons==0.16.1",
        "tensorflow-probability==0.14.1",
        "typing-extensions==3.7.4.3",
        "tf-agents==0.9.0",
        "pygame",
        "protobuf==3.19.6",
        "keras==2.6.0",
        "Cython==0.29.37",
        "mujoco_py==2.1.2.14",
        "dm_control==1.0.17",
        "mjrl @ git+https://github.com/aravindr93/mjrl@master#egg=mjrl"
    ]
)
