# CCOIP Library

CCOIP (Collective Communication Over IP) is a distributed communication library that provides efficient collective operations over IP networks. It supports both C++ and Python interfaces, making it suitable for various distributed computing applications including machine learning workloads.

## Prerequisites

- CMake (3.22.1 or higher)
- C++ compiler with C++11 support
- Python 3.x (for Python bindings)
- libuv
- Google Test (for running tests)

## Installation

### Building from Source (C++)

```bash
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make
```

### Installing Python Package

```bash
git submodule update --init --recursive
pip install python/framework
```

## Project Structure

- `ccoip/`: Core C++ library implementation
  - `public_include/`: Public API headers
  - `internal/`: Internal implementation details
  - `src/`: Source files
  - `tests/`: Unit and end-to-end tests
- `python/`: Python bindings and examples
  - `framework/`: Python package implementation
  - `tests/`: Python tests including MNIST distributed training example

## Testing

### C++ Tests
```bash
cd build
ctest
```

### Python Tests
```bash
cd python/tests
pip install -r requirements.txt
python -m pytest
```

## Examples

The library includes several examples:
- Basic reduction operations
- Distributed MNIST training using PyTorch DDP
- Network peer acceptance tests

For detailed examples, see:
- C++: `tests/basic_reduce_test/`
- Python: `python/tests/end_to_end/`

## License

TBD

## Contributing

TBD
