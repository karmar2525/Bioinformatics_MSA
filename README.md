# Sequence Alignment Toolkit with GUI

This Python application provides tools for biological sequence alignment, including global pairwise alignment (Needleman–Wunsch) and multiple sequence alignment (Center Star method). It features a graphical user interface, DNA sequence validation, alignment matrix visualization, and automatic PDF report generation.

---

## Features

- **Global pairwise alignment** using the Needleman–Wunsch algorithm  
- **Multiple sequence alignment** via the Center Star method  
- Input sequences from **FASTA files** or **manual text entry**  
- Validation of DNA sequences (**A, C, G, T** only)  
- Computation of alignment statistics: matches, mismatches, gaps, identity  
- Visualization of alignment matrices and MSA results  
- Auto-generated PDF reports with detailed summaries and parameters  
- GUI for easy interaction without command-line use  

---

## Installation

Install the required dependencies using pip:
         pip install -r requirements.txt

---
## Usage

1. Launch the application
    Run the program from your IDE or via command line:
    python MSA.py

2. Load DNA sequences
    - Upload a FASTA file via the GUI
    - Or enter sequences manually (only A, C, G, T allowed)

3. Set alignment parameters (optional)
    Adjust match, mismatch, and gap penalties as desired.

4. Run the alignment
    Click the "Run Alignment" button to perform the alignment.

5. View results
    - Alignment results and statistics appear in the GUI
    - A detailed PDF report is automatically generated and saved locally
