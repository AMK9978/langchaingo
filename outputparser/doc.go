/*
Package outputparser provides a set of output parsers to process structured or
unstructured data from language models (LLMs).

The outputparser package includes the following parsers:

  - Simple: a basic parser that returns the raw text as-is without any processing.
  - Structured: a parser that expects a JSON-formatted response and returns it as
    a map[string]string while validating against a provided schema.
  - Combining: a parser that combines the output of multiple parsers into a single parser.
  - CommaSeparatedList: a parser that takes a string with comma-separated values
    and returns them as a string slice.
  - RegexParser: a parser that takes a string, compiles it into a regular expression,
    and returns map[string]string of the regex groups.
  - RegexDict: a pareser that searches a string for values in a dictionary format,
    and returns a map[string]string of the keys and their associated value.
*/
package outputparser
