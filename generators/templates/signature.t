% if binding is None: # no binding -> print a signature
<%
# we want to align the parameter names so we need to know the length of the longest type name
longest_type_len = max(len(v) for v in sig.params.values())
tot_indent = len(sig.rt) + len(sig.attribute) + len(sig.name) + indent + 2 + (1 if len(sig.attribute) else 0)
%>\
${f"{sig.rt:>{len(sig.rt)+indent}} {sig.attribute}{' ' if len(sig.attribute) else ''}{sig.name}"}(
% for i, (n, t) in enumerate(sig.params.items()):
${(" " * tot_indent) + f"{t:<{longest_type_len}}"} ${n}${"," if i != sig.n_params-1 else ")"}
% endfor
% else: # we have a binding -> we are printing a function call
<%
tot_indent = len(sig.name) + indent + 1
%>\
${f"{sig.name:>{tot_indent-1}}"}(\
% for i, k in enumerate(sig.params):
${(" " * (tot_indent if i != 0 else 0)) + binding[k]}${"," if i != sig.n_params-1 else ");"}
% endfor
% endif
