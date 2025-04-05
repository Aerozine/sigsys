### install depedencies and generate plots
the code is configured to install all depedencies (from the code/Project.toml)
and execute the graph generator with thread=auto
to run it you can simply do from the root folder : 

```bash
make code
```

### generating report
to generate the report and generate all the plots required you can simply do
```bash
make report
```
> this will execute `make code` in order to retrieve the graphs
