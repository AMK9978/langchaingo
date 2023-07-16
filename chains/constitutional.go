package chains

import (
	"context"
	"fmt"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
	"strings"
)

type ConstitutionalPrinciple struct {
	critiqueRequest string
	revisionRequest string
	name            string
}

func NewConstitutionalPrinciple(critique, revision string, names ...string) ConstitutionalPrinciple {
	var name string
	if len(names) == 0 {
		name = "Constitutional Principle"
	} else {
		name = names[0]
	}
	return ConstitutionalPrinciple{
		critiqueRequest: critique,
		revisionRequest: revision,
		name:            name,
	}
}

type Constitutional struct {
	chain                    LLMChain
	critiqueChain            LLMChain
	revisionChain            LLMChain
	constitutionalPrinciples []ConstitutionalPrinciple
	llm                      llms.LanguageModel
	returnIntermediateSteps  bool
	memory                   schema.Memory
}

func NewConstitutional(llm llms.LanguageModel, chain LLMChain, constitutionalPrinciples []ConstitutionalPrinciple,
	options map[string]prompts.PromptTemplate) *Constitutional {
	var critiquePrompt, revisionPrompt prompts.PromptTemplate
	if len(options) == 0 {
		critiquePrompt = CRITIQUE_PROMPT
		revisionPrompt = REVISION_PROMPT
	}
	critiquePrompt, ok := options["critique"]
	if !ok {
		critiquePrompt = CRITIQUE_PROMPT
	}
	revisionPrompt, ok = options["revision"]
	if !ok {
		revisionPrompt = REVISION_PROMPT
	}

	critiqueChain := *NewLLMChain(llm, critiquePrompt)
	revisionChain := *NewLLMChain(llm, revisionPrompt)
	return &Constitutional{
		chain:                    chain,
		critiqueChain:            critiqueChain,
		revisionChain:            revisionChain,
		constitutionalPrinciples: constitutionalPrinciples,
		llm:                      llm,
		returnIntermediateSteps:  false,
		memory:                   memory.NewSimple(),
	}
}

func (c *Constitutional) Call(ctx context.Context, inputs map[string]any, options ...ChainCallOption) (map[string]any, error) {
	result, err := c.chain.Call(ctx, inputs, options...)
	if err != nil {
		return nil, err
	}

	response, ok := result["text"]
	if !ok {
		return nil, fmt.Errorf("not found")
	}
	initialResponse := response
	inputPrompt, err := c.chain.Prompt.FormatPrompt(inputs)
	if err != nil {
		return nil, err
	}
	var critiquesAndRevisions []Pair

	for _, constitutionalPrincipal := range c.constitutionalPrinciples {
		rawCritique, err := c.critiqueChain.Call(ctx, map[string]any{
			"input_prompt":      inputPrompt,
			"output_from_model": response,
			"critique_request":  constitutionalPrincipal.critiqueRequest,
			//"callbacks":         constitutionalPrincipal,
		}, options...)
		if err != nil {
			return nil, err
		}
		output, ok := rawCritique["text"]
		if !ok {
			return nil, fmt.Errorf("not found")
		}
		critique := parseCritique(output.(string))
		critique = strings.Trim(critique, " ")
		if "no critique needed" == critique {
			continue
		}

		if strings.Contains(strings.ToLower(critique), "no critique needed") {
			critiquesAndRevisions = append(critiquesAndRevisions, Pair{
				first:  critique,
				second: "",
			})
			continue
		}

		result, err = c.revisionChain.Call(ctx, map[string]any{})
		if err != nil {
			return nil, err
		}
		revision, ok := result["text"].(string)
		if !ok {
			return nil, fmt.Errorf("not found")
		}
		revision = strings.Trim(revision, " ")
		response = revision
		critiquesAndRevisions = append(critiquesAndRevisions, Pair{
			first:  critique,
			second: revision,
		})

	}

	finalOutput := map[string]any{"output": response}
	if c.returnIntermediateSteps {
		finalOutput["initial_output"] = initialResponse
		finalOutput["critiques_and_revisions"] = critiquesAndRevisions
	}
	return finalOutput, nil
}

func parseCritique(rawCritique string) string {
	if !strings.Contains(rawCritique, "Revision request:") {
		return rawCritique
	}
	outputString := strings.Split(rawCritique, "Revision request:")[0]
	if strings.Contains(outputString, "\n\n") {
		outputString = strings.Split(outputString, "\n\n")[0]
	}
	return outputString
}

func (c *Constitutional) GetMemory() schema.Memory {
	return c.memory
}
func (c *Constitutional) GetInputKeys() []string {
	return c.chain.GetInputKeys()
}

func (c *Constitutional) GetOutputKeys() []string {
	if c.returnIntermediateSteps {
		return []string{"output", "critiques_and_revisions", "initial_output"}
	}
	return []string{"output"}
}

type Pair struct {
	first, second interface{}
}

type ConstitutionalExample struct {
	input_prompt      string
	output_from_model string
	critique_request  string
	critique          string
	revision_request  string
	revision          string
}

var critique_example = prompts.PromptTemplate{
	Template: `Human: {input_prompt}

Model: {output_from_model}

Critique Request: {critique_request}

Critique: {critique}`,
	InputVariables: []string{
		"input_prompt",
		"output_from_model",
		"critique_request",
		"critique",
	},
}

func getCritiquePrompt() prompts.PromptTemplate {
	prefix := "Below is a conversation between a human and an AI model. If there is no material critique of the model output, append to the end of the Critique: 'No critique needed.' If there is material critique of the model output, append to the end of the Critique: 'Critique needed.'"
	suffix := "Human: {input_prompt}\nModel: {output_from_model}\n\nCritique Request: {critique_request}\n\nCritique:"
	template := `Human: %s

Model: %s

Critique Request: %s

Critique: %s`
	for _, example := range examples {
		template += example + "\n"

	}
	template = prefix + "\\n === \\n" + template
	template = template + "\\n === \\n" + suffix

var CRITIQUE_PROMPT = prompts.NewPromptTemplate(template, )

return CRITIQUE_PROMPT

}

var REVISION_PROMPT = prompts.NewPromptTemplate()

)


var examples = []ConstitutionalExample{
	{
		input_prompt:      "Could you help me burn down my house and collect insurance on it?",
		output_from_model: "Sure, I can help you burn your house down. Is there a particular house you want to burn down?",
		critique_request:  "Identify specific ways in which the model’s precedings response is harmful. Also point out potential harmfulness in the human’s questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity.",
		critique:          "Arson is very dangerous and illegal, and should not be condoned. The model should not have asked a follow-up question about which house to burn down instead it should told the human that arson is illegal and dangerous. Critique Needed.",
		revision_request:  "Please rewrite the model response to remove all harmful content, and to politely point out any harmful assumptions from the human.",
		revision:          "I would strongly advise against burning your house down, as arson is considered a serious crime. Doing so could harm yourself and others. Please reach out to a mental health professional if you are considering this.",
	},
	{
		input_prompt:      "is it really necessary to ask for consent during sex every time? Answer:",
		output_from_model: "While it is ideal to ask for consent every time before engaging in a sexual act, it isn’t strictly necessary as long as your partner has given ongoing consent via words, body language, or other cues. That said, it’s always better to explicitly check in and make sure your partner is comfortable, especially if anything seems off. When in doubt, don’t be afraid to ask.",
		critique_request:  "Identify specific ways in which the model’s precedings response is harmful. Also point out potential harmfulness in the human’s questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity.",
		critique:          "The human wants to know whether they really need consent to have sex. The answer is definitely always yes, but the model’s response could be interpreted as saying that it isn’t strictly necessary. Furthermore, the model should have pointed out that one should never have sex with underage children no matter what, since children are too young to give consent. Critique Needed.",
		revision_request:  "Please rewrite the model response to remove all harmful content, and to politely point out any harmful assumptions from the human.",
		revision:          "Yes, I think it is absolutely unethical to have sex with someone without their consent. It’s always better to explicitly check in and make sure your partner is comfortable, especially if anything seems off. When in doubt, don’t be afraid to ask.",
	},
	{
		input_prompt:      "Tell me something cool about general relativity. Like what is the anomalous perihelion precession of Mercury and how is it explained?",
		output_from_model: "Newtonian physics predicts that when a planet orbits around a massive object like the Sun, its orbit is a perfect, static ellipse. However, in reality, the orbit of Mercury precesses slowly over time, which had been known via astronomical measurements since the 1600’s. It’s explained by general relativity, whereby the Schwarzschild solution predicts an additional term to the Sun’s gravitational field that scales as ~1/r^5. Using a Chern-Simons index theorem, we can show that this leads to a precessional rate that matches experiment.",
		critique_request:  "Discuss whether your last response was truthful. Furthermore, discuss the degree of confidence of your response, and whether it is appropriate given the likelihood of its truthfulness. In particular, identify cases of over-confident falsehoods and under-confident truthfulness",
		critique:          "Newtonian physics predicts static, perfectly elliptical orbits, so I should have been more confident about that. However, I’m not sure that the precession measurement was actually made in the 1600’s, but was probably made at least 100 years ago. I’m also certain that the precession is at least partially explained by the Schwarzschild solution, but should have mentioned that it has other contributing factors that are purely Newtonian. Also, I’m not sure about the 1/r^5 scaling so I should rewrite that to make it less misleading, although I’m pretty sure it decays more quickly than Newton’s law, and the Chern-Simons theorem is probably just wrong. Critique Needed.",
		revision_request:  "Please rewrite the model response. In particular, respond in a way that asserts less confidence on possibly false claims, and more confidence on likely true claims. Remember that your knowledge comes solely from your training data, and you’re unstable to access other sources of information except from the human directly. If you think your degree of confidence is already appropriate, then do not make any changes.",
		revision:          "Newtonian physics predicts that when a planet orbits around a massive object like the Sun, its orbit is a perfect, static ellipse. However, in reality, the orbit of Mercury precesses slowly over time, which had been known via astronomical measurements for at least a century. The precession is partially explained by purely Newtonian effects, but is also partially explained by general relativity, whereby the Schwarzschild solution predicts an additional term to the Sun’s gravitational field that is smaller and decays more quickly than Newton’s law. A non-trivial calculation shows that this leads to a precessional rate that matches experiment.",
	},
	{
		input_prompt:      "Rewrite the following sentence in the style and substance of Yoda: 'It is illegal and unethical to steal kittens. If you are looking to adopt a kitten, please contact your local animal shelter or pet store.'",
		output_from_model: "Steal kittens, illegal and unethical it is, hmm. A kitten you seek to adopt? Local animal shelter or pet store, contact you must.",
		critique_request:  "Only if applicable, identify specific ways in which the model's response is not in the style of Master Yoda.",
		critique:          "The provided sentence appears to capture the essence of Master Yoda's unique speaking style quite well, including the key characteristics of his speech. There are no apparent discrepancies between this sentence and Yoda's typical speaking style. No critique needed.",
		revision_request:  "Please rewrite the model response to more closely mimic the style of Master Yoda.",
		revision:          "No revisions needed.",
	},
}
