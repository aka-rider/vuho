//! Print the names of all available audio input devices.
//!
//! `cargo run -p vuho-audio --example list_devices`

fn main() {
    match vuho_audio::list_input_device_names() {
        Ok(names) if names.is_empty() => println!("No input devices found."),
        Ok(names) => {
            println!("Input devices:");
            for name in names {
                println!("  - {name}");
            }
        }
        Err(e) => {
            eprintln!("Failed to list input devices: {e}");
            std::process::exit(1);
        }
    }
}
