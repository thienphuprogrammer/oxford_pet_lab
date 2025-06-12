from training.trainer import Trainer
from config.config import Config
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Oxford Pet Lab Training Script')
    parser.add_argument('--model_type', type=str, default='detection',
                       choices=['detection', 'segmentation', 'multitask'],
                       help='Type of model to train')
    parser.add_argument('--model_name', type=str, default='resnet50',
                       choices=['resnet50', 'mobilenetv2'],
                       help='Backbone model name')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(
        model_type=args.model_type,
        model_name=args.model_name,
        pretrained=args.pretrained
    )
    
    # Train model
    print(f"\nStarting training for {args.model_type}_{args.model_name}...")
    history = trainer.train(args.epochs)
    
    # Evaluate model
    print("\nEvaluating model...")
    trainer.evaluate()
    
    # Save model
    trainer.save_model()

if __name__ == '__main__':
    main()
