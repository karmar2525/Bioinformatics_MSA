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
    return bool(re.fullmatch(r"[ACGTacgt]+", seq))

def validate_sequence(seq):
    valid_chars = {'A', 'C', 'G', 'T'}
    return all(char in valid_chars for char in seq.upper())

def read_sequences(file_path=None, manual_input=False, seq_list=None):
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
    matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    for i in range(len_a + 1):
        matrix[i, 0] = i * gap_penalty
    for j in range(len_b + 1):
        matrix[0, j] = j * gap_penalty
    return matrix

def score_nw(a, b, match=1, mismatch=-1):
    return match if a == b else mismatch

def fill_nw_matrix(seq_a, seq_b, gap_penalty, match, mismatch):
    matrix = initialize_nw_matrix(len(seq_a), len(seq_b), gap_penalty)
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            match_score = matrix[i - 1, j - 1] + score_nw(seq_a[i - 1], seq_b[j - 1], match, mismatch)
            delete_score = matrix[i - 1, j] + gap_penalty
            insert_score = matrix[i, j - 1] + gap_penalty
            matrix[i, j] = max(match_score, delete_score, insert_score)
    return matrix

def traceback_nw(seq_a, seq_b, matrix, gap_penalty, match, mismatch):
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
    plt.savefig("nw_alignment_plot.png", dpi=300)
    plt.show()

def get_nw_path_coords(aligned_a, aligned_b):
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
    matrix = np.zeros((len_a + 1, len_b + 1), dtype=int)
    for i in range(len_a + 1):
        matrix[i][0] = i * gap_penalty
    for j in range(len_b + 1):
        matrix[0][j] = j * gap_penalty
    return matrix

def compute_msa_scores(a, b, matrix, match_score, mismatch_score, gap_penalty):
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            score = match_score if a[i - 1] == b[j - 1] else mismatch_score
            matrix[i][j] = max(
                matrix[i - 1][j] + gap_penalty,
                matrix[i][j - 1] + gap_penalty,
                matrix[i - 1][j - 1] + score
            )

def traceback_msa(a, b, matrix, match_score, mismatch_score, gap_penalty):
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
    matrix = initialize_msa_matrix(len(a), len(b), gap_penalty)
    compute_msa_scores(a, b, matrix, match_score, mismatch_score, gap_penalty)
    aligned_a, aligned_b = traceback_msa(a, b, matrix, match_score, mismatch_score, gap_penalty)
    return aligned_a, aligned_b

def center_star_msa(sequences, match_score=1, mismatch_score=-1, gap_penalty=-1):
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
    fig, ax = plt.subplots(figsize=(len(msa[0]) * 0.5, len(msa) * 0.5)) # Dostosuj rozmiar
    data = np.zeros((len(msa), len(msa[0])), dtype=int)
    cmap = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    color_map = ['lightgreen', 'lightblue', 'khaki', 'salmon', 'whitesmoke']
    for i, seq in enumerate(msa):
        for j, base in enumerate(seq):
            data[i, j] = cmap.get(base.upper(), 4) # Domyślnie gap

    im = ax.imshow(data, cmap=plt.cm.colors.ListedColormap(color_map), aspect='auto', interpolation='nearest')

    ax.set_xticks(np.arange(len(msa[0])))
    ax.set_yticks(np.arange(len(msa)))
    ax.set_xticklabels(list(msa[0])) # Możesz chcieć bardziej złożone labelowanie
    ax.set_yticklabels([f"Seq {i+1}" for i in range(len(msa))])
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_pdf_report(sequences, alignment=None, msa=None, params=None, nw_matrix=None, nw_path_coords=None, msa_stats=None):
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
                os.remove("msa.png")
            else:
                pdf.cell(0, 9, txt="Error: Could not generate MSA grid image.", ln=True)

            pdf.output("alignment_report.pdf")

def save_alignment_to_text_file(filename, sequences, alignment=None, msa=None, params=None, msa_stats=None):
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
    def __init__(self, root):
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
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            position_top = int(screen_height / 2 - height / 2)
            position_left = int(screen_width / 2 - width / 2)
            self.root.geometry(f'{width}x{height}+{position_left}+{position_top}')

    def upload_file(self):
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
        try:
            match = int(self.match_score.get())
            mismatch = int(self.mismatch_penalty.get())
            gap = int(self.gap_penalty.get())
            return {'match': match, 'mismatch': mismatch, 'gap': gap}
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for scoring parameters.")
            return None

    def run_alignment_data(self):
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
                return None
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return None

    def run_alignment_pdf(self):
        result_data = self.run_alignment_data()
        if result_data:
            sequences, alignment_result, params, nw_matrix, nw_path_coords, msa_stats = result_data
            if len(sequences) == 2:
                generate_pdf_report(sequences, alignment_result, None, params, nw_matrix, nw_path_coords, None)
                messagebox.showinfo("Success", "Pairwise alignment completed and PDF report saved.")
            elif len(sequences) > 2:
                generate_pdf_report(sequences, None, alignment_result, params, None, None, msa_stats)
                messagebox.showinfo("Success", "Multiple sequence alignment completed and PDF report saved.")
            else:
                messagebox.showerror("Error", "Please enter at least two sequences.")

    def save_alignment_text(self):
        result_data = self.run_alignment_data()
        if result_data:
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

    def clear_fields(self):
        self.seq_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.params_label.config(text="")

def main ():
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
                    msa_result = center_star_msa(sequences, args.match, args.mismatch, args.gap)
                    msa_stats = calculate_msa_statistics(msa_result)
                    print("Center Star Multiple Sequence Alignment:")
                    for seq in msa_result:
                        print(seq)
                    generate_pdf_report(sequences, msa=msa_result, params=params, msa_stats=msa_stats)
                    if args.output_text:
                        save_alignment_to_text_file(args.output_text, sequences, msa=msa_result, params=params,
                                                    msa_stats=msa_stats)

        except ValueError as e:
            print(f"Error: {e}")
        except FileNotFoundError:
            print("Error: FASTA file not found.")

if __name__ == "__main__":
    main()