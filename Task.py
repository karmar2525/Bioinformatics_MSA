"""
This script helps with DNA sequences. It can check if a sequence is valid,
read sequences from files or typed input, and get them ready for more analysis.
It also uses tools for handling commands, text patterns, math, making charts,
creating PDFs, reading biology files, and building simple user interfaces.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from Bio import SeqIO
from reportlab.lib import colors
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox

def is_valid_dna(seq):
    """
    Checks if a piece of text is a valid DNA sequence.
    This means it only contains the letters A, C, G, or T,
    no matter if they are uppercase or lowercase.

    Args:
        seq (str): The text to check.

    Returns:
        bool: True if it's valid DNA, False otherwise.
    """
    return bool(re.fullmatch(r"[ACGTacgt]+", seq))

def validate_sequence(seq):
    """
    Makes sure a sequence only has DNA letters (A, C, G, T).
    It doesn't care about uppercase or lowercase.

    Args:
        seq (str): The sequence to check.

    Returns:
        bool: True if all letters are A, C, G, or T; False otherwise.
    """
    valid_chars = {'A', 'C', 'G', 'T'}
    return all(char in valid_chars for char in seq.upper())

def read_sequences(file_path=None, manual_input=False, seq_list=None):
    """
    Gets DNA sequences from a FASTA file, a list you give it, or by asking you to type them in.
    All sequences are made uppercase and checked to make sure they are valid DNA.

    Args:
        file_path (str, optional): The path to a FASTA file. (e.g., "my_sequences.fasta").
                                   Leave empty if not using a file.
        manual_input (bool, optional): Set to True if you want to type sequences yourself.
                                       Leave as False otherwise.
        seq_list (list of str, optional): A list of DNA sequences as text (e.g., ["ATGC", "CGTA"]).
                                          Leave empty if not using a list.

    Returns:
        list of str: A list of DNA sequences that have been checked and are all uppercase.

    Raises:
        ValueError: If you don't tell it where to get sequences, if sequences have wrong letters,
                    if the file is empty, or if there aren't enough sequences (at least two needed).
        Exception: If something goes wrong while trying to read the file.
    """
    sequences = []
    if file_path:
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                seq = str(record.seq).upper()
                if validate_sequence(seq):
                    sequences.append(seq)
                else:
                    raise ValueError(f"Invalid sequence in FASTA file: {record.id}")
            if not sequences:
                raise ValueError("No valid sequences found in FASTA file.")
        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            raise
    elif seq_list:
        for seq in seq_list:
            seq_upper = seq.strip().upper()
            if validate_sequence(seq_upper):
                sequences.append(seq_upper)
            else:
                raise ValueError(f"Invalid sequence provided: {seq}")
        if len(sequences) < 2:
            raise ValueError("At least two valid sequences must be provided.")
    elif manual_input:
        print("Please enter the sequences manually.")
        while len(sequences) < 2:
            seq = input(f"Sequence {len(sequences) + 1}: ").strip().upper()
            if validate_sequence(seq):
                sequences.append(seq)
            else:
                print("Invalid sequence. Sequences must only contain the characters A, C, G, T.")
        return sequences
    else:
        raise ValueError("No sequence source provided.")


    if len(sequences) < 2 and not manual_input:
        raise ValueError("At least two sequences are required for alignment.")
    return sequences

def initialize_nw_matrix(len_a, len_b, gap_penalty):
    """
    Sets up the first matrix for Needleman-Wunsch alignment.
    It fills the first row and column with gap penalties.

    Args:
        len_a (int): The length of the first sequence.
        len_b (int): The length of the second sequence.
        gap_penalty (int): The penalty for a gap.

    Returns:
        numpy.ndarray: A matrix filled with zeros, with the first row and column
                       initialized with gap penalties.
    """
    matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    for i in range(len_a + 1):
        matrix[i, 0] = i * gap_penalty
    for j in range(len_b + 1):
        matrix[0, j] = j * gap_penalty
    return matrix

def score_nw(a, b, match=1, mismatch=-1):
    """
    Calculates the score for comparing two characters in Needleman-Wunsch.
    It gives a 'match' score if they are the same, and a 'mismatch' score if different.

    Args:
        a (str): The first character.
        b (str): The second character.
        match (int, optional): Score for a match. Defaults to 1.
        mismatch (int, optional): Score for a mismatch. Defaults to -1.

    Returns:
        int: The score (match or mismatch).
    """
    return match if a == b else mismatch

def fill_nw_matrix(seq_a, seq_b, gap_penalty, match, mismatch):
    """
    Fills in the Needleman-Wunsch score matrix.
    It calculates scores for matching, deleting, or inserting characters
    and picks the best (highest) score for each cell.

    Args:
        seq_a (str): The first sequence.
        seq_b (str): The second sequence.
        gap_penalty (int): The penalty for a gap.
        match (int): Score for a match.
        mismatch (int): Score for a mismatch.

    Returns:
        numpy.ndarray: The completed Needleman-Wunsch score matrix.
    """
    matrix = initialize_nw_matrix(len(seq_a), len(seq_b), gap_penalty)
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            match_score = matrix[i - 1, j - 1] + score_nw(seq_a[i - 1], seq_b[j - 1], match, mismatch)
            delete_score = matrix[i - 1, j] + gap_penalty
            insert_score = matrix[i, j - 1] + gap_penalty
            matrix[i, j] = max(match_score, delete_score, insert_score)
    return matrix

def traceback_nw(seq_a, seq_b, matrix, gap_penalty, match, mismatch):
    """
    Finds the best alignment path through the Needleman-Wunsch matrix.
    It starts from the bottom-right corner and moves back to the top-left,
    reconstructing the aligned sequences.

    Args:
        seq_a (str): The first original sequence.
        seq_b (str): The second original sequence.
        matrix (numpy.ndarray): The filled Needleman-Wunsch score matrix.
        gap_penalty (int): The penalty for a gap.
        match (int): Score for a match.
        mismatch (int): Score for a mismatch.

    Returns:
        tuple: A tuple containing two strings:
               - aligned_a (str): The first aligned sequence.
               - aligned_b (str): The second aligned sequence.
    """
    aligned_a = ""
    aligned_b = ""
    i, j = len(seq_a), len(seq_b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and matrix[i, j] == matrix[i - 1, j - 1] + score_nw(seq_a[i - 1], seq_b[j - 1], match, mismatch):
            aligned_a = seq_a[i - 1] + aligned_a
            aligned_b = seq_b[j - 1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and matrix[i, j] == matrix[i - 1, j] + gap_penalty:
            aligned_a = seq_a[i - 1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        else:
            aligned_a = "-" + aligned_a
            aligned_b = seq_b[j - 1] + aligned_b
            j -= 1
    return aligned_a, aligned_b

def plot_nw_matrix(matrix, seq_a, seq_b, path_coords, title='Needleman-Wunsch Score Matrix', alignment_path_text=''):
    """
    Creates and saves a visual plot of the Needleman-Wunsch score matrix.
    It shows the scores in each cell and highlights the optimal alignment path.

    Args:
        matrix (numpy.ndarray): The Needleman-Wunsch score matrix.
        seq_a (str): The first sequence.
        seq_b (str): The second sequence.
        path_coords (list of tuple): A list of (row, column) tuples representing
                                     the coordinates of the optimal path.
        title (str, optional): The title for the plot. Defaults to 'Needleman-Wunsch Score Matrix'.
        alignment_path_text (str, optional): Additional text to display below the plot,
                                             often showing the aligned sequences. Defaults to ''.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score", rotation=270, labelpad=15)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{int(matrix[i, j])}', ha='center', va='center', color='black', fontsize=8)
    ax.set_xticks(np.arange(len(seq_b) + 1))
    ax.set_yticks(np.arange(len(seq_a) + 1))
    ax.set_xticklabels([' '] + list(seq_b))
    ax.set_yticklabels([' '] + list(seq_a))
    ax.set_xlabel("Sequence B")
    ax.set_ylabel("Sequence A")

    if path_coords:
        path_y, path_x = zip(*path_coords)
        ax.plot(path_x, path_y, color='deeppink', linewidth=2, marker='o', markersize=4, label="Optimal Path")
        ax.legend(loc='upper left')
    plt.title(title)
    ax.text(0.5, -0.15, alignment_path_text, ha='center', va='top', transform=ax.transAxes, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("nw_matrix.png", dpi=300)
    plt.show()

def get_nw_path_coords(aligned_a, aligned_b):
    """
    Calculates the coordinates of the optimal alignment path on the Needleman-Wunsch matrix.
    This is used for plotting the path.

    Args:
        aligned_a (str): The first aligned sequence (with gaps).
        aligned_b (str): The second aligned sequence (with gaps).

    Returns:
        list of tuple: A list of (row, column) coordinates representing the path.
    """
    coords = [(0, 0)]
    x, y = 0, 0
    for a, b in zip(aligned_a, aligned_b):
        if a != '-' and b != '-':
            x += 1
            y += 1
        elif a != '-':
            x += 1
        elif b != '-':
            y += 1
        coords.append((x, y))
    return coords

def initialize_msa_matrix(len_a, len_b, gap_penalty):
    """
    Sets up the first matrix for Multiple Sequence Alignment (MSA).
    It fills the first row and column with gap penalties, similar to NW.

    Args:
        len_a (int): The length of the first sequence.
        len_b (int): The length of the second sequence.
        gap_penalty (int): The penalty for a gap.

    Returns:
        numpy.ndarray: A matrix filled with zeros, with the first row and column
                       initialized with gap penalties.
    """
    matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    for i in range(len_a + 1):
        matrix[i][0] = i * gap_penalty
    for j in range(len_b + 1):
        matrix[0][j] = j * gap_penalty
    return matrix

def compute_msa_scores(a, b, matrix, match_score, mismatch_score, gap_penalty):
    """
    Calculates scores for a Multiple Sequence Alignment (MSA) matrix.
    This function seems to be a part of filling a matrix for pairwise alignment
    within a larger MSA context, similar to the Needleman-Wunsch filling logic.

    Args:
        a (str): The first sequence.
        b (str): The second sequence.
        matrix (numpy.ndarray): The matrix to fill (already initialized).
        match_score (int): Score for matching characters.
        mismatch_score (int): Score for mismatched characters.
        gap_penalty (int): Penalty for introducing a gap.

    Returns:
        None: The function modifies the `matrix` in place.
    """
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            score = match_score if a[i - 1] == b[j - 1] else mismatch_score
            matrix[i][j] = max(
                matrix[i - 1][j] + gap_penalty,
                matrix[i][j - 1] + gap_penalty,
                matrix[i - 1][j - 1] + score
            )

def traceback_msa(a, b, matrix, match_score, mismatch_score, gap_penalty):
    """
    Performs a traceback for Multiple Sequence Alignment (MSA) to reconstruct aligned sequences.
    This is similar to Needleman-Wunsch traceback but adapted for MSA context.

    Args:
        a (str): The first sequence.
        b (str): The second sequence.
        matrix (numpy.ndarray): The filled MSA score matrix.
        match_score (int): Score for a match.
        mismatch_score (int): Score for a mismatch.
        gap_penalty (int): The penalty for a gap.

    Returns:
        tuple: A tuple containing two strings:
               - aligned_a (str): The first aligned sequence.
               - aligned_b (str): The second aligned sequence.
    """
    aligned_a, aligned_b = "", ""
    i, j = len(a), len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and matrix[i][j] == matrix[i - 1][j - 1] + (match_score if a[i - 1] == b[j - 1] else mismatch_score):
            aligned_a = a[i - 1] + aligned_a
            aligned_b = b[j - 1] + aligned_b
            i -= 1
            j -= 1
        elif i > 0 and matrix[i][j] == matrix[i - 1][j] + gap_penalty:
            aligned_a = a[i - 1] + aligned_a
            aligned_b = "-" + aligned_b
            i -= 1
        else:
            aligned_a = "-" + aligned_a
            aligned_b = b[j - 1] + aligned_b
            j -= 1
    return aligned_a, aligned_b

def needleman_wunsch_msa(a, b, match_score=1, mismatch_score=-1, gap_penalty=-1):
    """
    Performs a Needleman-Wunsch global alignment between two sequences.
    This function combines matrix initialization, filling, and traceback.

    Args:
        a (str): The first sequence.
        b (str): The second sequence.
        match_score (int, optional): Score for a match. Defaults to 1.
        mismatch_score (int, optional): Score for a mismatch. Defaults to -1.
        gap_penalty (int, optional): The penalty for a gap. Defaults to -1.

    Returns:
        tuple: A tuple containing two strings:
               - aligned_a (str): The first aligned sequence.
               - aligned_b (str): The second aligned sequence.
    """
    matrix = initialize_msa_matrix(len(a), len(b), gap_penalty)
    compute_msa_scores(a, b, matrix, match_score, mismatch_score, gap_penalty)
    aligned_a, aligned_b = traceback_msa(a, b, matrix, match_score, mismatch_score, gap_penalty)
    return aligned_a, aligned_b

def center_star_msa(sequences, match_score=1, mismatch_score=-1, gap_penalty=-1):
    """
    Performs Multiple Sequence Alignment (MSA) using the Center Star method.
    It finds a center sequence and then aligns all other sequences to it.

    Args:
        sequences (list of str): A list of DNA sequences to align.
        match_score (int, optional): Score for a match. Defaults to 1.
        mismatch_score (int, optional): Score for a mismatch. Defaults to -1.
        gap_penalty (int, optional): The penalty for a gap. Defaults to -1.

    Returns:
        list of str: A list of aligned sequences.
    """
    n = len(sequences)
    if n < 2:
        return sequences
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            aligned_a, aligned_b = needleman_wunsch_msa(sequences[i], sequences[j], match_score, mismatch_score, gap_penalty)
            score = sum([match_score if a == b else mismatch_score if a != '-' and b != '-' else gap_penalty
                         for a, b in zip(aligned_a, aligned_b)])
            distance_matrix[i][j] = distance_matrix[j][i] = -score
    center_index = np.argmin(np.sum(distance_matrix, axis=0))
    center_seq = sequences[center_index]
    msa = [center_seq]

    for i in range(n):
        if i == center_index:
            continue
        aligned_center, aligned_seq = needleman_wunsch_msa(center_seq, sequences[i], match_score, mismatch_score, gap_penalty)
        msa = merge_alignments(msa, aligned_center, aligned_seq)
    return msa

def merge_alignments(msa, aligned_center, aligned_seq):
    """
    Merges a new pairwise alignment (aligned_center, aligned_seq) into an existing MSA.
    It adds gaps to the existing MSA sequences to match the length of the new alignment.

    Args:
        msa (list of str): The current list of aligned sequences.
        aligned_center (str): The center sequence from the new pairwise alignment.
        aligned_seq (str): The other sequence from the new pairwise alignment.

    Returns:
        list of str: The updated list of aligned sequences after merging.
    """
    new_msa = []
    for seq in msa:
        new_seq = ""
        index = 0
        for char in aligned_center:
            if char == '-':
                new_seq += '-'
            else:
                new_seq += seq[index]
                index += 1
        new_msa.append(new_seq)
    new_msa.append(aligned_seq)

    max_len = max(len(seq) for seq in new_msa)
    new_msa = [seq.ljust(max_len, '-') for seq in new_msa]
    return new_msa


def calculate_msa_statistics(msa):
    """
    Calculates statistics for a Multiple Sequence Alignment (MSA),
    including matches, mismatches, gaps, and overall identity.

    Args:
        msa (list of str): The list of aligned sequences.

    Returns:
        tuple: A tuple containing:
               - matches (int): Number of perfectly matched columns.
               - mismatches (int): Number of mismatched columns (no gaps, but not all same).
               - gaps (int): Total number of gaps across all sequences.
               - identity (float): Percentage of identical columns.
    """
    if not msa:
        return 0, 0, 0, 0.0
    matches = 0
    mismatches = 0
    gaps = 0
    length = len(msa[0])
    num_seq = len(msa)

    for i in range(length):
        column = [seq[i] for seq in msa]
        if '-' in column:
            gaps += column.count('-')
        elif all(base == column[0] for base in column):
            matches += 1
        else:
            mismatches += 1
    identity = (matches / length) * 100 if length > 0 else 0.0
    return matches, mismatches, gaps, identity

def draw_alignment_grid(c, msa, start_x, start_y, cell_size=12):
    """
    Draws a colored grid representation of a Multiple Sequence Alignment (MSA)
    onto a ReportLab Canvas object. Each base is colored differently.

    Args:
        c (reportlab.pdfgen.canvas.Canvas): The ReportLab Canvas object to draw on.
        msa (list of str): The list of aligned sequences.
        start_x (float): The starting X coordinate for the grid.
        start_y (float): The starting Y coordinate for the grid.
        cell_size (int, optional): The size of each cell in the grid. Defaults to 12.
    """
    base_colors = {
        'A': colors.lightgreen,
        'C': colors.lightblue,
        'G': colors.khaki,
        'T': colors.salmon,
        '-': colors.whitesmoke
    }
    rows = len(msa)
    cols = len(msa[0])
    for row in range(rows):
        for col in range(cols):
            base = msa[row][col]
            color = base_colors.get(base.upper(), colors.white)
            x = start_x + col * cell_size
            y = start_y - row * cell_size
            c.setFillColor(color)
            c.rect(x, y, cell_size, cell_size, fill=1, stroke=0)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + cell_size / 2, y + 2, base)

def draw_alignment_matplotlib(msa, filename="msa_grid.png"):
    """
    Draws a colored grid representation of a Multiple Sequence Alignment (MSA)
    using Matplotlib and saves it as an image file.

    Args:
        msa (list of str): The list of aligned sequences.
        filename (str, optional): The name of the file to save the image.
                                  Defaults to "msa_grid.png".
    """
    fig, ax = plt.subplots(figsize=(len(msa[0]) * 0.5, len(msa) * 0.5))
    data = np.zeros((len(msa), len(msa[0])), dtype=int)
    cmap = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    color_map = ['lightgreen', 'lightblue', 'khaki', 'salmon', 'whitesmoke']
    for i, seq in enumerate(msa):
        for j, base in enumerate(seq):
            data[i, j] = cmap.get(base.upper(), 4)

    im = ax.imshow(data, cmap=plt.cm.colors.ListedColormap(color_map), aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(len(msa[0])))
    ax.set_yticks(np.arange(len(msa)))
    ax.set_xticklabels(list(msa[0]))
    ax.set_yticklabels([f"Seq {i+1}" for i in range(len(msa))])
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_pdf_report(sequences, alignment=None, msa=None, params=None, nw_matrix=None, nw_path_coords=None, msa_stats=None):
    """
    Generates a PDF report containing alignment results, parameters,
    and visualizations (Needleman-Wunsch matrix plot or MSA grid).

    Args:
        sequences (list of str): The original input sequences.
        alignment (tuple, optional): A tuple (aligned_a, aligned_b) for pairwise alignment. Defaults to None.
        msa (list of str, optional): A list of aligned sequences for MSA. Defaults to None.
        params (dict, optional): A dictionary of alignment parameters (match, mismatch, gap). Defaults to None.
        nw_matrix (numpy.ndarray, optional): The Needleman-Wunsch score matrix. Defaults to None.
        nw_path_coords (list of tuple, optional): Coordinates of the NW optimal path. Defaults to None.
        msa_stats (tuple, optional): Statistics for MSA (matches, mismatches, gaps, identity). Defaults to None.

    Returns:
        FPDF: The FPDF object representing the generated PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(150, 10, txt="Sequence Alignment Report", ln=True, align='C')
    pdf.ln(2)

    if params:
        pdf.set_font("Arial", style='B', size=9)
        pdf.cell(0, 9, txt="Alignment Parameters:", ln=True)
        pdf.set_font("Arial", size=9)
        for key, value in params.items():
            pdf.cell(0, 8, txt=f"{key.capitalize()}: {value}", ln=True)
        pdf.ln(2)

    pdf.set_font("Arial", size=9)
    pdf.cell(0, 9, txt=f"Number of Sequences: {len(sequences)}", ln=True)
    for i, seq in enumerate(sequences):
        pdf.cell(0, 9, txt=f"Sequence {i + 1}: {seq}", ln=True)

    pdf.ln(2)
    if alignment:
        seq_a, seq_b = sequences[:2]
        aligned_a, aligned_b = alignment
        matches = sum(1 for a, b in zip(aligned_a, aligned_b) if a == b)
        gaps = aligned_a.count('-') + aligned_b.count('-')
        identity = (matches / len(aligned_a)) * 100 if len(aligned_a) > 0 else 0.0

        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(0, 9, txt="Needleman-Wunsch Global Alignment:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 9, txt=f"Sequence A: {seq_a}", ln=True)
        pdf.cell(0, 9, txt=f"Sequence B: {seq_b}", ln=True)
        pdf.cell(0, 9, txt=f"Aligned A: {aligned_a}", ln=True)
        pdf.cell(0, 9, txt=f"Aligned B: {aligned_b}", ln=True)
        pdf.cell(0, 9, txt=f"Matches: {matches}", ln=True)
        pdf.cell(0, 9, txt=f"Gaps: {gaps}", ln=True)
        pdf.cell(0, 9, txt=f"Identity: {identity:.2f}%", ln=True)
        pdf.ln(2)

        if nw_matrix is not None and nw_path_coords:
            plt.figure(figsize=(9, 8))
            plt.imshow(nw_matrix, cmap='viridis')
            plt.colorbar(label="Score")
            plt.xticks(np.arange(len(seq_b) + 1), [' '] + list(seq_b))
            plt.yticks(np.arange(len(seq_a) + 1), [' '] + list(seq_a))
            path_y, path_x = zip(*nw_path_coords)
            plt.plot(path_x, path_y, color='deeppink', linewidth=2, marker='o', markersize=4, label="Optimal Path")
            plt.title(f"Needleman-Wunsch Score Matrix (Match: {params.get('match', 1)}, Mismatch: {params.get('mismatch', -1)}, Gap: {params.get('gap', -1)})")
            plt.xlabel("Sequence B")
            plt.ylabel("Sequence A")
            plt.legend()
            plt.tight_layout()
            plt.savefig("nw_matrix.png")
            pdf.image("nw_matrix.png", x=10, y=pdf.get_y(), w=100)
            pdf.ln(90)
            os.remove("nw_matrix.png")
        pdf.output("alignment_report.pdf")

    elif msa:
        pdf.set_font("Arial", style='B', size=9)
        pdf.cell(0, 9, txt="Center Star Multiple Sequence Alignment:", ln=True)
        pdf.set_font("Arial", size=9)
        for seq in msa:
            pdf.cell(0, 8, txt=seq, ln=True)
        pdf.ln(2)
        if msa_stats:
            matches, mismatches, gaps, identity = msa_stats
            pdf.set_font("Arial", style='B', size=9)
            pdf.cell(0, 9, txt="MSA Statistics:", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 9, txt=f"Matches (in columns): {matches}", ln=True)
            pdf.cell(0, 9, txt=f"Mismatches (in columns): {mismatches}", ln=True)
            pdf.cell(0, 9, txt=f"Gaps: {gaps}", ln=True)
            pdf.cell(0, 9, txt=f"Identity (average column): {identity:.2f}%", ln=True)
            pdf.ln(2)

            pdf.set_font("Arial", style='B', size=9)
            pdf.cell(0, 9, txt="Colored Multiple Sequence Alignment Grid:", ln=True)

            draw_alignment_matplotlib(msa, "msa.png")

            if os.path.exists("msa.png"):
                pdf.image("msa.png", x=10, y=pdf.get_y(), w=70)
                os.remove("msa.png") # Clean up temporary file
            else:
                pdf.cell(0, 9, txt="Error: Could not generate MSA grid image.", ln=True)

            pdf.output("alignment_report.pdf")

    return pdf

def save_alignment_to_text_file(filename, sequences, alignment=None, msa=None, params=None, msa_stats=None):
    """
    Saves alignment results (pairwise or MSA) to a text file.

    Args:
        filename (str): The path to the text file where results will be saved.
        sequences (list of str): The original input sequences.
        alignment (tuple, optional): A tuple (aligned_a, aligned_b) for pairwise alignment. Defaults to None.
        msa (list of str, optional): A list of aligned sequences for MSA. Defaults to None.
        params (dict, optional): A dictionary of alignment parameters. Defaults to None.
        msa_stats (tuple, optional): Statistics for MSA. Defaults to None.
    """
    with open(filename, "w") as f:
        if params:
            f.write("Alignment Parameters:\n")
            for key, value in params.items():
                f.write(f"{key.capitalize()}: {value}\n")
            f.write("\n")

        f.write(f"Number of Sequences: {len(sequences)}\n")
        for i, seq in enumerate(sequences):
            f.write(f"Sequence {i+1}: {seq}\n")
        f.write("\n")

        if alignment:
            aligned_a, aligned_b = alignment
            f.write("Needleman-Wunsch Global Alignment:\n")
            f.write(f"Sequence A: {aligned_a}\n")
            f.write(f"Sequence B: {aligned_b}\n")
            matches = sum(1 for a, b in zip(aligned_a, aligned_b) if a == b)
            gaps = aligned_a.count('-') + aligned_b.count('-')
            identity = (matches / len(aligned_a)) * 100 if len(aligned_a) > 0 else 0.0
            f.write(f"Matches: {matches}\n")
            f.write(f"Gaps: {gaps}\n")
            f.write(f"Identity: {identity:.2f}%\n")
        elif msa:
            f.write("Center Star Multiple Sequence Alignment:\n")
            for seq in msa:
                f.write(f"{seq}\n")
            f.write("\n")
            if msa_stats:
                matches, mismatches, gaps, identity = msa_stats
                f.write("MSA Statistics:\n")
                f.write(f"Matches (in columns): {matches}\n")
                f.write(f"Mismatches (in columns): {mismatches}\n")
                f.write(f"Gaps: {gaps}\n")
                f.write(f"Identity (average column): {identity:.2f}%\n")
    print(f"Alignment saved to {filename}")

class AlignmentApp:
    """
    A Tkinter-based graphical user interface (GUI) application for
    performing DNA sequence alignment (pairwise Needleman-Wunsch or
    Center Star Multiple Sequence Alignment).
    """
    def __init__(self, root):
        """
        Initializes the AlignmentApp GUI.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("Multiple Sequence Alignment")
        self.center_window(700, 700)

        notebook = ttk.Notebook(root)
        notebook.grid(row=0, column=0, padx=10, pady=10)

        self.input_tab = ttk.Frame(notebook)
        self.param_tab = ttk.Frame(notebook)
        self.result_tab = ttk.Frame(notebook)

        notebook.add(self.input_tab, text="Input")
        notebook.add(self.param_tab, text="Parameters")
        notebook.add(self.result_tab, text="Results")

        tk.Label(self.input_tab, text="Enter DNA sequences (one per line):", font=("Arial", 11)).pack(anchor="w",
                                                                                                      padx=10,
                                                                                                      pady=5)

        self.seq_text = tk.Text(self.input_tab, height=12, width=70, bg="#e0f7fa", fg="black", font=("Consolas", 10))
        self.seq_text.pack(padx=10, pady=5)

        self.upload_button = tk.Button(self.input_tab, text="📂 Upload FASTA File", command=self.upload_file,
                                       bg="#00796b", fg="black")
        self.upload_button.pack(pady=10)


        self.match_score = tk.IntVar(value=1)
        self.mismatch_penalty = tk.IntVar(value=-1)
        self.gap_penalty = tk.IntVar(value=-1)

        tk.Label(self.param_tab, text="Match Score:", font=("Arial", 11)).grid(row=0, column=0, padx=10, pady=10,
                                                                               sticky="w")
        self.match_score_spin = tk.Spinbox(self.param_tab, from_=-10, to=10, textvariable=self.match_score,
                                           width=10)
        self.match_score_spin.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self.param_tab, text="Mismatch Penalty:", font=("Arial", 11)).grid(row=1, column=0, padx=10,
                                                                                    pady=10, sticky="w")
        self.mismatch_penalty_spin = tk.Spinbox(self.param_tab, from_=-10, to=10,
                                                textvariable=self.mismatch_penalty, width=10)
        self.mismatch_penalty_spin.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self.param_tab, text="Gap Penalty:", font=("Arial", 11)).grid(row=2, column=0, padx=10, pady=10,
                                                                               sticky="w")
        self.gap_penalty_spin = tk.Spinbox(self.param_tab, from_=-10, to=10, textvariable=self.gap_penalty,
                                           width=10)
        self.gap_penalty_spin.grid(row=2, column=1, padx=10, pady=10)


        tk.Label(self.result_tab, text="Alignment Result:", font=("Arial", 11)).pack(anchor="w", padx=10, pady=5)

        self.result_text = tk.Text(self.result_tab, height=15, width=80, bg="#f1f8e9", fg="black",
                                   font=("Courier", 10))
        self.result_text.pack(padx=10, pady=5)

        button_frame = tk.Frame(self.result_tab)
        button_frame.pack(pady=10)

        self.run_pdf_button = tk.Button(button_frame, text="Run & Save PDF", command=self.run_alignment_pdf,
                                        bg="lightpink", fg="black")
        self.run_pdf_button.grid(row=0, column=0, padx=5)

        self.save_text_button = tk.Button(button_frame, text="Save as Text", command=self.save_alignment_text,
                                          bg="lightpink", fg="black")
        self.save_text_button.grid(row=0, column=1, padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_fields, bg="lightpink",
                                      fg="black")
        self.clear_button.grid(row=0, column=2, padx=5)

        self.params_label = ttk.Label(self.result_tab, text="", justify='left', font=("Arial", 10))
        self.params_label.pack(anchor="w", padx=10, pady=5)

    def center_window(self, width, height):
        """
        Centers the Tkinter window on the screen.

        Args:
            width (int): The desired width of the window.
            height (int): The desired height of the window.
        """
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_top = int(screen_height / 2 - height / 2)
        position_left = int(screen_width / 2 - width / 2)
        self.root.geometry(f'{width}x{height}+{position_left}+{position_top}')

    def upload_file(self):
        """
        Opens a file dialog to allow the user to select a FASTA file.
        Reads sequences from the selected file and displays them in the text area.
        Shows an error if the file is invalid or contains fewer than two sequences.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("FASTA files", ("*.fasta", "*.fa")), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    sequences = [str(record.seq) for record in SeqIO.parse(f, "fasta")]
                    if len(sequences) < 2:
                        raise ValueError("FASTA file must contain at least two sequences.")
                    self.seq_text.delete(1.0, tk.END)
                    self.seq_text.insert(tk.END, "\n".join(sequences))
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading the file: {e}")

    def get_alignment_parameters(self):
        """
        Retrieves the alignment parameters (match score, mismatch penalty, gap penalty)
        from the GUI input fields.

        Returns:
            dict: A dictionary containing 'match', 'mismatch', and 'gap' scores,
                  or None if input values are invalid.
        """
        try:
            match = int(self.match_score.get())
            mismatch = int(self.mismatch_penalty.get())
            gap = int(self.gap_penalty.get())
            return {'match': match, 'mismatch': mismatch, 'gap': gap}
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for scoring parameters.")
            return None

    def run_alignment_data(self):
        """
        Executes the sequence alignment based on user input and parameters.
        Performs Needleman-Wunsch for two sequences or Center Star MSA for more.
        Displays results in the GUI and generates a plot for Needleman-Wunsch.

        Returns:
            tuple: A tuple containing (sequences, alignment_result, params, nw_matrix, nw_path_coords, msa_stats).
                   Returns None if there's an error or insufficient sequences.
        """
        seqs = self.seq_text.get("1.0", tk.END).strip().splitlines()
        params = self.get_alignment_parameters()
        if not params:
            return None, None, None, None, None, None
        try:
            sequences = read_sequences(seq_list=seqs)
            match_score = params['match']
            mismatch_penalty = params['mismatch']
            gap_penalty = params['gap']

            if len(sequences) == 2:
                aligned_a, aligned_b = needleman_wunsch_msa(sequences[0], sequences[1], match_score, mismatch_penalty, gap_penalty)
                matches = sum(1 for a, b in zip(aligned_a, aligned_b) if a == b)
                gaps = aligned_a.count('-') + aligned_b.count('-')
                identity = (matches / len(aligned_a)) * 100 if len(aligned_a) > 0 else 0.0
                result_str = f"Needleman-Wunsch Alignment:\nSequence A: {aligned_a}\nSequence B: {aligned_b}\nMatches: {matches}\nGaps: {gaps}\nIdentity: {identity:.2f}%\n"
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, result_str)
                self.params_label.config(text=f"Match: {match_score}, Mismatch: {mismatch_penalty}, Gap: {gap_penalty}")
                nw_matrix = fill_nw_matrix(sequences[0], sequences[1], gap_penalty, match_score, mismatch_penalty)
                path_coords = get_nw_path_coords(aligned_a, aligned_b)
                plot_nw_matrix(nw_matrix, sequences[0], sequences[1], path_coords, title=f"NW Matrix (Match: {match_score}, Mismatch: {mismatch_penalty}, Gap: {gap_penalty})")
                return sequences, (aligned_a, aligned_b), params, nw_matrix, path_coords, None

            elif len(sequences) > 2:
                msa = center_star_msa(sequences, match_score, mismatch_penalty, gap_penalty)
                matches, mismatches, gaps, identity = calculate_msa_statistics(msa)
                result_str = "Center Star Multiple Sequence Alignment:\n" + "\n".join(msa) + f"\n\nMatches (in columns): {matches}\nMismatches (in columns): {mismatches}\nGaps: {gaps}\nIdentity (average column): {identity:.2f}%\n"
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, result_str)
                self.params_label.config(text=f"Match: {match_score}, Mismatch: {mismatch_penalty}, Gap: {gap_penalty}")
                return sequences, msa, params, None, None, (matches, mismatches, gaps, identity)
            else:
                messagebox.showerror("Error", "Please enter at least two sequences.")
                return None, None, None, None, None, None
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return None, None, None, None, None, None

    def run_alignment_pdf(self):
        """
        Runs the alignment and then generates and saves a PDF report of the results.
        """
        result_data = self.run_alignment_data()
        if result_data and result_data[0] is not None: # Check if sequences are not None
            sequences, alignment_result, params, nw_matrix, nw_path_coords, msa_stats = result_data
            if len(sequences) == 2:
                pdf_report = generate_pdf_report(sequences, alignment_result, None, params, nw_matrix, nw_path_coords, None)
                messagebox.showinfo("Success", "Pairwise alignment completed and PDF report saved.")
            elif len(sequences) > 2:
                pdf_report = generate_pdf_report(sequences, None, alignment_result, params, None, None, msa_stats)
                messagebox.showinfo("Success", "Multiple sequence alignment completed and PDF report saved.")
            else:
                messagebox.showerror("Error", "Please enter at least two sequences.")
        else:
            messagebox.showerror("Error", "No valid alignment data to generate PDF.")


    def save_alignment_text(self):
        """
        Runs the alignment and then saves the results to a text file.
        """
        result_data = self.run_alignment_data()
        if result_data and result_data[0] is not None:
            sequences, alignment, params, _, _, msa_stats = result_data
            file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                     filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if file_path:
                if len(sequences) == 2:
                    save_alignment_to_text_file(file_path, sequences, alignment=alignment, params=params)
                else:
                    save_alignment_to_text_file(file_path, sequences, msa=alignment if len(sequences) > 2 else None,
                                                params=params, msa_stats=msa_stats)
                messagebox.showinfo("Success", f"Alignment saved to {file_path}")
        else:
            messagebox.showerror("Error", "No valid alignment data to save as text.")

    def clear_fields(self):
        """
        Clears all input and output fields in the GUI.
        """
        self.seq_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.params_label.config(text="")

def main ():
    """
    Main function to parse command-line arguments and run the alignment tool.
    It can run in GUI mode, pairwise alignment mode, or MSA mode.
    """
    parser = argparse.ArgumentParser(description="Perform pairwise or multiple sequence alignment.")
    parser.add_argument('--mode', choices=['pairwise', 'msa', 'gui'], default='gui',
                        help="Choose the alignment mode: pairwise (Needleman-Wunsch), msa (Center Star), or gui (graphical interface). Default is gui.")
    parser.add_argument('--file', help="Path to a FASTA file containing sequences.")
    parser.add_argument('--seqs', nargs='+', help="List of sequences for alignment (for pairwise or MSA).")
    parser.add_argument('--match', type=int, default=1, help="Match score.")
    parser.add_argument('--mismatch', type=int, default=-1, help="Mismatch penalty.")
    parser.add_argument('--gap', type=int, default=-1, help="Gap penalty.")
    parser.add_argument('--output_text', help="Path to save the alignment in text format.")
    args = parser.parse_args()

    try:
        if args.mode == 'gui':
            root = tk.Tk()
            app = AlignmentApp(root)
            root.mainloop()
        else:
            sequences = read_sequences(file_path=args.file, seq_list=args.seqs)
            params = {'match': args.match, 'mismatch': args.mismatch, 'gap': args.gap}

            if args.mode == 'pairwise':
                if len(sequences) < 2:
                    print("Error: At least two sequences are required for pairwise alignment.")
                    return
                seq_a, seq_b = sequences[:2]
                nw_matrix = fill_nw_matrix(seq_a, seq_b, args.gap, args.match, args.mismatch)
                aligned_a, aligned_b = traceback_nw(seq_a, seq_b, nw_matrix, args.gap, args.match, args.mismatch)
                path_coords = get_nw_path_coords(aligned_a, aligned_b)
                print("Needleman-Wunsch Alignment:")
                print(f"Sequence A: {aligned_a}")
                print(f"Sequence B: {aligned_b}")
                generate_pdf_report(sequences[:2], alignment=(aligned_a, aligned_b), params=params,
                                    nw_matrix=nw_matrix, nw_path_coords=path_coords)
                plot_nw_matrix(nw_matrix, seq_a, seq_b, path_coords,
                               title=f"NW Matrix (Match: {args.match}, Mismatch: {args.mismatch}, Gap: {args.gap})")
                if args.output_text:
                    save_alignment_to_text_file(args.output_text, sequences[:2], alignment=(aligned_a, aligned_b),
                                                params=params)

            elif args.mode == 'msa':
                if len(sequences) < 2:
                    print("Error: At least two sequences are required for multiple sequence alignment.")
                    return
                msa = center_star_msa(sequences, args.match, args.mismatch, args.gap)
                matches, mismatches, gaps, identity = calculate_msa_statistics(msa)
                print("Center Star Multiple Sequence Alignment:")
                for seq in msa:
                    print(seq)
                print(f"\nMatches (in columns): {matches}")
                print(f"Mismatches (in columns): {mismatches}")
                print(f"Gaps: {gaps}")
                print(f"Identity (average column): {identity:.2f}%")
                generate_pdf_report(sequences, None, msa, params, None, None, (matches, mismatches, gaps, identity))
                if args.output_text:
                    save_alignment_to_text_file(args.output_text, sequences, msa=msa, params=params,
                                                msa_stats=(matches, mismatches, gaps, identity))
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
