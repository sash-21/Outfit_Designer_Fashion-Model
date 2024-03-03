if epoch % 10 == 0:
    # Sample images for each class/subclass
    labels = torch.arange(args.num_classes).long().to(device)
    sampled_images = diffusion.sample(model, n=args.num_classes, labels=labels)
    ema_sampled_images = diffusion.sample(ema_model, n=args.num_classes, labels=labels)
    
    # Plot and save sampled images
    plot_images(sampled_images)
    save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
    save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
    
    # Save model checkpoints
    torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
    torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_{epoch}.pt"))
    torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))
