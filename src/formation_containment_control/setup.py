import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'formation_containment_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.py'))),
        # Include config files (yaml and rviz)
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*.yaml')) + 
            glob(os.path.join('config', '*.rviz'))),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Robotarium Team',
    maintainer_email='robotarium@example.com',
    description='Formation-Containment Control with Adaptive Sliding Mode Strategy',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulation_node.py = scripts.simulation_node:main',
            'virtual_leader_node.py = scripts.virtual_leader_node:main',
            'formation_containment_node.py = scripts.formation_containment_node:main',
            'visualization_node.py = scripts.visualization_node:main',
            'standalone_demo.py = scripts.standalone_demo:main',
            'crazyswarm_bridge_node.py = scripts.crazyswarm_bridge_node:main',
        ],
    },
)

