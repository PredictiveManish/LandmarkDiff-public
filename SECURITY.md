# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in LandmarkDiff, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

You have two options:

1. **GitHub Security Advisories** (preferred): Use [GitHub's private vulnerability reporting](https://github.com/dreamlessx/LandmarkDiff-public/security/advisories/new).
2. **Email**: Contact dreamlessx directly through GitHub.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any affected versions
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Patch for critical issues**: Within 30 days
- **Patch for non-critical issues**: Within 90 days
- **Public disclosure**: Coordinated with reporter after fix is available

## Security Scope

### In scope

- **Injection vulnerabilities**: Code injection through crafted inputs to the inference pipeline or CLI
- **Path traversal**: Unauthorized file access through manipulated file paths in config loading or image I/O
- **Credential exposure**: Accidental leaking of API keys, tokens, or user data through logs or error messages
- **Code execution**: Arbitrary code execution through model deserialization (pickle loads) or config parsing
- **Data leakage**: Unintended exposure of input images or intermediate representations
- **Dependency vulnerabilities**: Known CVEs in direct dependencies that affect LandmarkDiff functionality

### Out of scope

- **Aesthetic quality issues**: Model output quality, artifacts, or unrealistic results are not security vulnerabilities
- **Upstream-only vulnerabilities**: Issues in dependencies that do not affect LandmarkDiff's usage (report to the upstream project)
- **Social engineering attacks**: Phishing or other social engineering targeting project maintainers or users
- **Local-only Gradio demo**: The Gradio demo has no authentication by default and is intended for local use only
- **Model bias or fairness**: While important, these are tracked separately from security vulnerabilities
- **Denial of service via large inputs**: Processing large images or many landmarks is expected to consume resources

## Security Best Practices for Users

- Never expose the Gradio demo to the public internet without authentication
- Rotate any API keys or tokens regularly
- Use the provided Apptainer/Docker containers for isolation
- Keep dependencies updated (`pip install -U landmarkdiff`)
- Avoid loading model weights or configs from untrusted sources
- Review YAML config files before use, as they can reference arbitrary file paths
