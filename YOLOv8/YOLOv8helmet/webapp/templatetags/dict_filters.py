from django import template

register = template.Library()


@register.filter
def get(mapping, key):
    try:
        return mapping[key]
    except Exception:
        return ''
