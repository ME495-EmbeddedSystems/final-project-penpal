"""Setup file for Penpal."""

from pathlib import Path

from setuptools import find_packages, setup


def recursive_files(prefix, path):
    """
    Recurse over path returning a list of tuples.

    :param prefix: prefix path to prepend to the path.
    :param path: Path to directory to recurse.
                 Path should not have a trailing '/'.
    :return: List of tuples.
             First element of each tuple is destination path.
             Second element is a list of files to copy to that path.

    """
    return [
        (
            str(Path(prefix) / subdir),
            [str(file) for file in subdir.glob('*') if not file.is_dir()],
        )
        for subdir in Path(path).glob('**')
    ]


package_name = 'penpal'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        *recursive_files('share/' + package_name, 'launch'),
        *recursive_files('share/' + package_name, 'config'),
    ],
    install_requires=[
        'modern_roboticsopencv-python',
        'setuptools',
        'transforms3d',
        *recursive_files('share/' + package_name, 'rviz'),
    ],
    zip_safe=True,
    maintainer='conorbot',
    maintainer_email='cwoodhayes@gmail.com',
    description='Robot that writes to a whiteboard held by a human.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'board_detector = penpal.nodes.board_detector:main',
            'int_test_ppcontrol = penpal.integration_tests.int_test_ppcontrol:main',
        ],
    },
)
