# Contributions

Contributions and pull requests are welcome.  Please take note of the following guidelines:

* **Work and push on [dev](https://github.com/kalray/metalibm/tree/dev) branch**. `git remote update; git checkout origin/dev` before hacking. Exceptions for bugs in `master` branch and simple typo fix.
* Always rebase your work before push: `git pull --rebase origin dev`
* Adhere to the existing style as much as possible; notably, tabulation indents and long-form keywords. Metalibm is trying to follow [PEP8](https://www.python.org/dev/peps/pep-0008/), at least for any new development. Try to add comments for important changes/addings.
* Make targeted pull request: propose one feature/bugfix, and push only commit(s) related to this feature/bugfix.
* Any push to your pull request will trigger a regression job on [gitlab](https://gitlab.com/nibrunie/metalibm/pipelines).

# Bugs

* To make it easier to reproduce, please supply the following:
  * the SHA1 of metalibm you're using (SHA1 of pythonsollya may also be interesting)
  * a step by step command to reproduce the bug (which meta-functions, with which options, at which stage ...)
