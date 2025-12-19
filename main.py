"""
Script principale per eseguire l'intero workflow.
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Crack Detection con Autoencoder')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'inference', 'evaluate', 'all'],
                       help='Modalità di esecuzione')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Threshold per anomaly detection')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n" + "="*60)
        print("FASE 1: TRAINING")
        print("="*60 + "\n")
        from train import train
        train()
    
    if args.mode == 'inference' or args.mode == 'all':
        print("\n" + "="*60)
        print("FASE 2: INFERENCE")
        print("="*60 + "\n")
        from inference import inference
        inference()
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n" + "="*60)
        print("FASE 3: EVALUATION")
        print("="*60 + "\n")
        from evaluate import evaluate_model, find_best_threshold
        
        # Valuta con threshold specificato
        evaluate_model(threshold=args.threshold)
        
        # Trova threshold ottimale
        find_best_threshold()
    
    print("\n" + "="*60)
    print("COMPLETATO!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
