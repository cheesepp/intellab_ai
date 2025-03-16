from schema.prompts import DevelopmentTemplates, OfficialTemplates


class PromptManager:
    def __init__(self, development_mode=False):
        self.development_mode = development_mode
        self.templates = DevelopmentTemplates.as_dict() if development_mode else OfficialTemplates.as_dict()
    
    def get_template(self, template_name):
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]
    
    def format_prompt(self, template_name, **kwargs):
        template = self.get_template(template_name)
        return template.format(**kwargs)
    
    def __getattr__(self, name):
        """Allow template access via attributes (e.g., manager.GLOBAL_CHATBOT_TEMPLATE)."""
        try:
            return self.get_template(name)
        except ValueError as e:
            # Convert ValueError to AttributeError for attribute access
            raise AttributeError(str(e)) from e