# Contributing to Local Summarizer

Thank you for your interest in contributing to Local Summarizer! This project aims to make private AI accessible to everyone.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Include your OS, Python version, and any error messages
- Describe steps to reproduce the issue

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test your changes locally
5. Commit with clear messages
6. Push and create a Pull Request

### Code Style

- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Keep code simple and readable
- Add comments for complex logic

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/local-summarizer.git
cd local-summarizer

# Install with dev dependencies
uv sync --dev

# Run linting
uv run ruff check .
```

## Areas for Contribution

- **Language Support**: Add prompts for more languages
- **Model Support**: Add support for other local models
- **UI Improvements**: Enhance the frontend interface
- **Documentation**: Improve docs and examples
- **Performance**: Optimize inference speed

## Questions?

Feel free to open an issue for discussion or reach out on [LinkedIn](https://www.linkedin.com/in/changyu-hu-607a55149/).

Thank you for helping make private AI more accessible!
