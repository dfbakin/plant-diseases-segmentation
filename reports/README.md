# Compiling Reports

This directory contains the reporting infrastructure for the project. The reports are built using [Quarto](https://quarto.org/) and LaTeX.

## How to Compile All Reports

### 1. Install Dependencies

Make sure you have installed all required Python dependencies as specified in the project. For example, using Poetry:

```bash
poetry install
```

You also need to have `xelatex` installed on your system. This is required for building PDF reports with proper formatting.

```bash
sudo apt-get install texlive-xetex
```

### 2. Render All Reports

From the `reports/` directory, run:

```bash
poetry run quarto render
```

This command will build all reports specified in this directory using the Quarto configuration, rendering them to the `_output` directory in both PDF and other specified formats.
