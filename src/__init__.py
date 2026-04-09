"""
ReviewGuard: Fake Review Detection on Yelp
==========================================

A two-branch hybrid fake review detection system combining:
  - Text Branch: Fine-tuned RoBERTa-base (768-dim [CLS] embeddings)
  - Behavior Branch: 6 handcrafted reviewer behavioral features
  - Fusion: Concatenation → 774-dim → 2-layer MLP with focal loss

Authors:
    Rithwik Reddy Donthi Reddy
    Rohan Mukka
"""

__version__ = "1.0.0"
__authors__ = ["Rithwik Reddy Donthi Reddy", "Rohan Mukka"]
