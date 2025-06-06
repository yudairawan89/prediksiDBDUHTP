streamlit.errors.StreamlitAPIException: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).

Traceback:
File "/mount/src/prediksidbduhtp/app.py", line 114, in <module>
    with st.expander(f"{row['kecamatan']} — Risiko: {row['Prediksi Risiko DBD']}"):
         ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 444, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/layouts.py", line 601, in expander
    return self.dg._block(block_proto=block_proto)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/delta_generator.py", line 522, in _block
    _check_nested_element_violation(self, block_type, ancestor_block_types)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/delta_generator.py", line 602, in _check_nested_element_violation
    raise StreamlitAPIException(
        "Expanders may not be nested inside other expanders."
    )
