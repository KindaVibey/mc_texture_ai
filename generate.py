import argparse
from pathlib import Path
from texture_generator import MinecraftTextureAI

def main():
    parser = argparse.ArgumentParser(
        description='Minecraft 16x16 Texture AI Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--train', type=str, choices=['blocks', 'items', 'all'],
                        help='Train the AI models on your texture data')
    
    parser.add_argument('--generate', nargs=3, metavar=('TYPE', 'CATEGORY', 'NAME'),
                        help='Generate a shaded texture. TYPE: blocks/items, CATEGORY: subfolder name, NAME: output filename')
    
    parser.add_argument('--colorize', nargs=3, metavar=('PATH', 'TYPE', 'CATEGORY'),
                        help='Colorize a shaded texture. PATH: path to shaded PNG, TYPE: blocks/items, CATEGORY: subfolder name')
    
    parser.add_argument('--auto-colorize', action='store_true',
                        help='Automatically colorize after generation (use with --generate)')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible generation')
    
    parser.add_argument('--list-categories', type=str, choices=['blocks', 'items'],
                        help='List available categories for blocks or items')
    
    parser.add_argument('--epochs-shading', type=int, default=500,
                        help='Number of training epochs for shading model (default: 500)')
    
    parser.add_argument('--epochs-color', type=int, default=300,
                        help='Number of training epochs for colorizer (default: 300)')
    
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    ai = MinecraftTextureAI(str(script_dir))

    if args.train:
        print("\n" + "="*70)
        print("TRAINING MODE")
        print("="*70)
        
        if args.train == 'all':
            types_to_train = ['blocks', 'items']
        else:
            types_to_train = [args.train]
        
        for texture_type in types_to_train:

            data_dir = script_dir / 'training_data' / texture_type
            if not data_dir.exists() or not any(data_dir.iterdir()):
                print(f"\n⚠ No training data found for {texture_type} in {data_dir}")
                print(f"   Please add PNG files in subfolders (e.g., {data_dir}/wood/, {data_dir}/metal/)")
                continue
            
            print(f"\nTraining {texture_type}...")
            ai.train_shading_model(texture_type, epochs=args.epochs_shading)
            ai.train_colorizer(texture_type, epochs=args.epochs_color)
        
        print("\n" + "="*70)
        print("✓ Training complete!")
        print("="*70)

    elif args.list_categories:
        texture_type = args.list_categories
        ai.load_models(texture_type)
        
        if texture_type in ai.category_maps:
            categories = list(ai.category_maps[texture_type].keys())
            print(f"\nAvailable {texture_type} categories:")
            for cat in categories:
                print(f"  • {cat}")
        else:
            print(f"⚠ No trained model found for {texture_type}. Train first with --train {texture_type}")

    elif args.generate:
        texture_type, category, name = args.generate
        
        print("\n" + "="*70)
        print(f"GENERATING {texture_type.upper()} TEXTURE")
        print("="*70)
        print(f"Category: {category}")
        print(f"Output: {name}")
        if args.seed:
            print(f"Seed: {args.seed}")
        print("="*70 + "\n")
        
        try:

            shaded_path = ai.generate_texture(texture_type, category, name, seed=args.seed)

            if args.auto_colorize:
                print("\nAuto-colorizing...")
                colored_path = ai.colorize_texture(shaded_path, texture_type, category, name)
                print("\n" + "="*70)
                print("✓ Generation complete!")
                print(f"  Shaded: {shaded_path}")
                print(f"  Colored: {colored_path}")
                print("="*70)
            else:
                print("\n" + "="*70)
                print("✓ Generation complete!")
                print(f"  Output: {shaded_path}")
                print(f"\nTo colorize, run:")
                print(f"  python generate.py --colorize {shaded_path} {texture_type} {category}")
                print("="*70)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTip: Use --list-categories to see available categories")

    elif args.colorize:
        path, texture_type, category = args.colorize
        
        print("\n" + "="*70)
        print("COLORIZING TEXTURE")
        print("="*70)
        print(f"Input: {path}")
        print(f"Category: {category}")
        print("="*70 + "\n")
        
        try:
            colored_path = ai.colorize_texture(path, texture_type, category)
            print("\n" + "="*70)
            print("✓ Colorization complete!")
            print(f"  Output: {colored_path}")
            print("="*70)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()