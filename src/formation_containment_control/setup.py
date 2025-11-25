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
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Robotarium Team',
    maintainer_email='robotarium@example.com',
    description='Formation-Containment Control with Adaptive Sliding Mode Strategy',
    license='MIT',
    tests_require=['pytest'],
)

