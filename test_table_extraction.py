#!/usr/bin/env python3
"""
Test script for PDF table extraction functionality

This script tests the enhanced PDF processor to verify:
1. Table detection and extraction
2. Markdown conversion
3. Structured data preservation
4. Metadata enrichment
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from processors.pdf_processor import PDFProcessor
import json


def test_pdf_processor():
    """Test the PDF processor with table extraction"""
    processor = PDFProcessor()

    print("=" * 70)
    print("PDF Table Extraction Test")
    print("=" * 70)
    print()

    # Check if there are any PDF files in the data directory
    data_path = Path('./data')
    if not data_path.exists():
        print("⚠️  No data directory found. Creating it...")
        data_path.mkdir(parents=True, exist_ok=True)
        print("✓ Created ./data directory")
        print()
        print("📝 Please add a PDF file with tables to ./data/ and run this test again.")
        return

    pdf_files = list(data_path.glob('**/*.pdf'))

    if not pdf_files:
        print("⚠️  No PDF files found in ./data/")
        print("📝 Please add a PDF file with tables to ./data/ and run this test again.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in ./data/")
    print()

    # Test the first PDF file
    test_file = pdf_files[0]
    print(f"Testing file: {test_file}")
    print("-" * 70)
    print()

    # Process the PDF
    try:
        chunks = processor.process(str(test_file))

        if not chunks:
            print("❌ No chunks extracted from PDF")
            return

        print(f"✓ Successfully extracted {len(chunks)} chunks")
        print()

        # Analyze chunks
        text_chunks = [c for c in chunks if c['metadata'].get('content_type') == 'text']
        table_chunks = [c for c in chunks if c['metadata'].get('content_type') == 'table']

        print("📊 Chunk Summary:")
        print(f"  - Text chunks: {len(text_chunks)}")
        print(f"  - Table chunks: {len(table_chunks)}")
        print()

        # Display table chunks in detail
        if table_chunks:
            print("=" * 70)
            print("TABLE CHUNKS FOUND")
            print("=" * 70)
            print()

            for i, chunk in enumerate(table_chunks, 1):
                metadata = chunk['metadata']
                print(f"Table {i}:")
                print(f"  Page: {metadata.get('page')}")
                print(f"  Table Number: {metadata.get('table_number')}")

                table_meta = metadata.get('table_metadata', {})
                print(f"  Dimensions: {table_meta.get('num_rows')} rows × {table_meta.get('num_columns')} columns")
                print(f"  Headers: {table_meta.get('headers')}")
                print(f"  Extraction Method: {table_meta.get('extraction_method')}")
                print()

                print("  Markdown Format:")
                print("  " + "-" * 66)
                lines = chunk['text'].split('\n')
                for line in lines[:10]:  # Show first 10 lines
                    print(f"  {line}")
                if len(lines) > 10:
                    remaining = len(lines) - 10
                    print(f"  ... ({remaining} more lines)")
                print()

                # Show structured data (first 3 rows)
                structured = metadata.get('structured_data', {})
                if structured.get('data'):
                    print("  Structured Data (JSON):")
                    print("  " + "-" * 66)
                    data = structured['data'][:3]  # First 3 rows
                    print(f"  {json.dumps(data, indent=4)}")
                    if len(structured['data']) > 3:
                        print(f"  ... ({len(structured['data']) - 3} more rows)")
                print()
                print("-" * 70)
                print()

        else:
            print("⚠️  No tables found in this PDF")
            print("   This could mean:")
            print("   - The PDF doesn't contain tables")
            print("   - The tables are in a format that's hard to detect")
            print("   - The tables are images (not text-based)")
            print()

        # Display first text chunk as sample
        if text_chunks:
            print("=" * 70)
            print("SAMPLE TEXT CHUNK")
            print("=" * 70)
            print()
            sample = text_chunks[0]
            print(f"Page: {sample['metadata'].get('page')}")
            print(f"Content Type: {sample['metadata'].get('content_type')}")
            print()
            print("Text Preview:")
            print("-" * 70)
            print(sample['text'][:500] + "..." if len(sample['text']) > 500 else sample['text'])
            print()

        print("=" * 70)
        print("✓ Test completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_pdf_processor()
