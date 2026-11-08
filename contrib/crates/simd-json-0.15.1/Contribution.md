# Contribution Guidelines for Rust Port of Extremely Fast simdjson JSON Parser with Serde Compatibility

Thank you for your interest in contributing to the Rust port of the simdjson JSON parser with serde compatibility. Your contributions are greatly appreciated. This document outlines the guidelines for contributing to the project to ensure a collaborative and productive development environment.

## Table of Contents

1. Getting Started
2. Code of Conduct
3. How to Contribute
    - Reporting Bugs
    - Adding Features
    - Code Contributions
4. Coding Guidelines
5. Testing
6. License

## Getting Started

Before you begin contributing, make sure you have:

- Rust and Cargo installed on your system.
- To take advantage of simd-json your system needs to be SIMD-capable
- A GitHub account for version control and issue tracking.
- Make sure you are using the newly released version
- Familiarize yourself with the project by reviewing the example in this repository `example` folder and understanding its goals.

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the expected behavior and conduct within the project's community. We expect respectful and professional interactions from all participants.

## How to Contribute

We welcome contributions in various forms:

### Reporting Bugs

If you discover any bugs, issues, or unexpected behavior in the Rust port, please report them by opening a new issue on the [GitHub Issues](https://github.com/simd-lite/simd-json/issues) page. Make sure to provide detailed information about the bug, including steps to reproduce it.

### Adding Features

If you have ideas for new features or improvements, please discuss them by opening a new issue on the [GitHub Issues](https://github.com/simd-lite/simd-json/issues) section. Engage with the Contributors on new features or improvements and refine your proposal.

### Code Contributions

For code contributions:

1. Fork the project repository on GitHub.
2. Clone your fork locally: `git clone https://github.com/your-username/your-repo.git`
3. Create a new branch for your changes: `git checkout -b feature/your-feature`
4. Write your code, following the Code of Conduct (see [Code of Conduct](#CODE_OF_CONDUCT.md)).
5. Write unit tests for your code (see [Tests](#tests)).
6. You can refer to `data` folder for further information (see [data](#data)) 
7. Push your changes to your fork on GitHub: `git push origin feature/your-feature`
8. Create a Pull Request (PR) in the project repository, providing a clear description of your changes and linking to any relevant issues or discussions.

## Coding Guidelines

To maintain a consistent and readable codebase, please follow these guidelines:

- Adhere to Rust's official style guide and best practices.
- Document your code using comments and provide clear explanations for complex logic.
- Aim for code that is efficient and idiomatic in Rust.
- Ensure your code is compatible with serde for JSON serialization/deserialization.

## Testing

We require comprehensive test coverage to maintain the quality of the codebase. Write unit tests for your code and ensure that existing tests pass. Use continuous integration tools to automatically run tests for various Rust versions.

## License

By contributing to this project, you agree that your contributions will be licensed under Apache License, Version 2.0, and MIT license. Please review the project's `LICENSE` file for more information.

Thank you for considering contributing to the Rust port of the simd-json . Your contributions are essential to the project's success and the Rust community as a whole.
