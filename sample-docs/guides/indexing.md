# Indexing Guide

## Chunking

Documents are split on markdown headings, then on length. Each chunk records the
1-based line span it occupies in the source file. That span must reproduce the
chunk text exactly when sliced back out; the test suite asserts this.

## Tokenization

Latin text is tokenized on word boundaries. CJK runs are emitted as character
bigrams plus unigrams, because whitespace tokenization returns nothing useful for
Chinese or Japanese.

## Cache

The index cache stores full chunk text. Never commit it.
