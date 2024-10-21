#convert all the plots to pdf
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

#function to convert plot to pdf, accepts fig object
def to_pdf(figures, filename='plots.pdf'):
    """
    Saves a list of matplotlib figure objects to a single PDF file,
    each figure on a separate page.

    Parameters:
    - figures: List of matplotlib.figure.Figure objects
    - filename: Name of the output PDF file
    """
    #assign folder to filename if churn is in filename
    
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

