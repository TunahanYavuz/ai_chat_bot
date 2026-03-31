use egui::{Align2, RichText, Vec2};

use crate::api::provider_models;
use crate::config::{ApiProvider, Settings, DEFAULT_MODEL_ID};

pub struct SetupWizard {
    step: usize,
    selected_provider: ApiProvider,
    api_key: String,
    base_url: String,
    default_model: String,
    working_directory: String,
    shell_execution_enabled: bool,
}

impl SetupWizard {
    pub fn new(settings: &Settings) -> Self {
        let provider = settings.selected_provider.clone();
        let default_model = if settings.default_model.is_empty() {
            provider_models(provider.key())
                .into_iter()
                .next()
                .map(|m| m.id)
                .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string())
        } else {
            settings.default_model.clone()
        };

        Self {
            step: 0,
            selected_provider: provider,
            api_key: settings.active_api_key(),
            base_url: settings.active_base_url(),
            default_model,
            working_directory: settings.working_directory.clone(),
            shell_execution_enabled: settings.shell_execution_enabled,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, settings: &mut Settings) -> bool {
        let mut completed = false;

        egui::Window::new("First-time setup")
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.set_min_width(600.0);
                ui.heading("Welcome to AI Chat Bot");
                ui.label(format!("Step {}/6", self.step + 1));
                ui.separator();

                match self.step {
                    0 => {
                        ui.label("This quick wizard configures your provider and workspace.");
                        ui.label("You can change these settings any time later.");
                    }
                    1 => {
                        ui.label("Choose your AI provider:");
                        egui::ComboBox::from_id_salt("setup_provider")
                            .selected_text(self.selected_provider.display_name())
                            .show_ui(ui, |ui| {
                                for provider in ApiProvider::all() {
                                    if ui
                                        .selectable_value(
                                            &mut self.selected_provider,
                                            provider.clone(),
                                            provider.display_name(),
                                        )
                                        .clicked()
                                    {
                                        self.base_url = provider.default_base_url().to_string();
                                        let models = provider_models(provider.key());
                                        if let Some(first) = models.first() {
                                            self.default_model = first.id.clone();
                                        }
                                    }
                                }
                            });
                        ui.label(self.selected_provider.description());
                    }
                    2 => {
                        ui.label("Enter API configuration:");
                        ui.horizontal(|ui| {
                            ui.label("API Key:");
                            ui.text_edit_singleline(&mut self.api_key);
                        });
                        ui.horizontal(|ui| {
                            ui.label("Base URL:");
                            ui.text_edit_singleline(&mut self.base_url);
                        });
                    }
                    3 => {
                        ui.label("Pick a default model:");
                        let models = provider_models(self.selected_provider.key());
                        egui::ComboBox::from_id_salt("setup_model")
                            .selected_text(self.default_model.clone())
                            .show_ui(ui, |ui| {
                                for model in models {
                                    ui.selectable_value(
                                        &mut self.default_model,
                                        model.id.clone(),
                                        model.name,
                                    );
                                }
                            });
                    }
                    4 => {
                        ui.label("Set your workspace directory:");
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.working_directory);
                            if ui.button("Browse").clicked() {
                                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                    self.working_directory = path.to_string_lossy().to_string();
                                }
                            }
                        });
                    }
                    _ => {
                        ui.label("Review configuration:");
                        ui.label(format!("Provider: {}", self.selected_provider.display_name()));
                        ui.label(format!("Base URL: {}", self.base_url));
                        ui.label(format!("Default model: {}", self.default_model));
                        ui.label(format!("Working directory: {}", self.working_directory));
                        ui.checkbox(
                            &mut self.shell_execution_enabled,
                            RichText::new("Allow shell execution"),
                        );
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.step > 0, egui::Button::new("Back"))
                        .clicked()
                    {
                        self.step -= 1;
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.step == 5 {
                            if ui.button("Finish").clicked() {
                                settings.selected_provider = self.selected_provider.clone();
                                settings.set_provider_config(
                                    &self.selected_provider,
                                    &self.api_key,
                                    &self.base_url,
                                );
                                settings.default_model = self.default_model.clone();
                                settings.working_directory = self.working_directory.clone();
                                settings.shell_execution_enabled = self.shell_execution_enabled;
                                settings.setup_complete = true;
                                completed = true;
                            }
                        } else if ui.button("Next").clicked() {
                            self.step += 1;
                        }
                    });
                });
            });

        completed
    }
}
