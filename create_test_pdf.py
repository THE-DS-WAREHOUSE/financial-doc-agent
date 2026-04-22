from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

lines = [
    "Q3 2024 Financial Report - Fundtech Corp",
    "",
    "Revenue",
    "Net revenue for Q3 2024 was $4.2M, compared to $3.7M in Q2 2024,",
    "representing a 13.5% increase quarter over quarter.",
    "",
    "Risk Factors",
    "Key risk factors include market volatility, regulatory changes in the",
    "fintech sector, and exposure to credit default risk in the SME segment.",
    "Total non-performing loans increased by 2.1% compared to last quarter.",
    "",
    "Debt and Equity",
    "Total debt stands at $12M with shareholder equity of $48M,",
    "resulting in a debt-to-equity ratio of 0.25.",
    "",
    "Fraud Detection",
    "Our fraud detection system flagged 320 high-risk transactions in Q3,",
    "reducing fraudulent exposure by 10% compared to Q2.",
    "",
    "Outlook",
    "We expect revenue growth of 8-10% in Q4 2024, driven by expansion",
    "into new markets and increased adoption of our KYB pipeline.",
]

for line in lines:
    pdf.cell(0, 10, line, ln=True)

pdf.output("docs/test_report.pdf")
print("PDF created at docs/test_report.pdf")