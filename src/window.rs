use std::sync::Arc;

use egui_winit::{winit, EventResponse};

#[cfg(target_arch = "wasm32")]
use winit::event_loop::{self};
use winit::{application::ApplicationHandler, dpi::{PhysicalSize, Size}, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::PhysicalKey, window::Window};

use egui_wgpu::wgpu;
use wgpu::*;

use crate::render::*;


pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>
}

impl App {

    #[allow(dead_code)]
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut, unused_assignments)]
        let mut window_attributes = Window::default_attributes();
                    
        #[cfg(not(target_arch = "wasm32"))]
        {
            window_attributes =
                Window::default_attributes()
                .with_inner_size(
                    Size::Physical(
                        PhysicalSize { 
                            width: 1800, 
                            height: 1200 
                        }
                    )
                );
        }  
            
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use wasm_bindgen::UnwrapThrowExt;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // not on web, use pollster to await. otherwise make it async
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(proxy
                        .send_event(
                            State::new(window)
                                .await
                                .expect("Unable to create canvas!")
                        )
                        .is_ok()
                    ) 
                });
            }
        }
    }


    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height
            );
        }
        self.state = Some(event);
    } 
    
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return
        };

        let consumed = state.egui_state.on_window_event(&state.window, &event);
        
        if consumed.consumed {
            if matches!(event, WindowEvent::Resized(_)) {

            }
            else if !matches!(event, WindowEvent::RedrawRequested) {
                state.window.request_redraw();
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => (),
                    Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    },
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            },

            WindowEvent::KeyboardInput { 
                event: KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(&event_loop, code, key_state.is_pressed()),

            WindowEvent::CursorMoved { 
                position,
                ..
            } => state.handle_mouse_moved(&event_loop, position),

            WindowEvent::MouseInput { 
                state: btn_state,
                button,
                ..
            } => {
                state.handle_mouse_event(button, btn_state.is_pressed());
            }

            _ => ()
        }
    }
}


#[allow(dead_code)]
pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::UnwrapThrowExt;
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}