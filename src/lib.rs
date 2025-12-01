mod window;
mod render;
mod shader_structs;
mod texture;
mod camera;
mod helper;
mod scene;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    use window::*;
    use wasm_bindgen::UnwrapThrowExt;


    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
