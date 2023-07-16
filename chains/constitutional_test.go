package chains

import (
	"context"
	"fmt"
	"github.com/tmc/langchaingo/prompts"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms/openai"
)

func TestConstitutionCritiqueParsing(t *testing.T) {
	TEXT_ONE := ` This text is bad.

	Revision request: Make it better.
	
	Revision:`

	TEXT_TWO := " This text is bad.\n\n"

	TEXT_THREE := ` This text is bad.
	
	Revision request: Make it better.
	
	Revision: Better text`

	for _, rawCritique := range []string{TEXT_ONE, TEXT_TWO, TEXT_THREE} {
		critique := parseCritique(rawCritique)

		require.Equal(t, strings.Trim(critique, " "), "This text is bad.",
			fmt.Sprintf("Failed on %s with %s", rawCritique, critique))
	}

}

func Test(t *testing.T) {
	t.Parallel()
	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	model, err := openai.New()
	require.NoError(t, err)
	chain := *NewLLMChain(model, prompts.NewPromptTemplate("{{.text}}", []string{"text"}))

	c := NewConstitutional(model, chain, []ConstitutionalPrinciple{NewConstitutionalPrinciple(
		"Tell if this answer is good.",
		"Give a better answer."),
	}, nil)
	_, err = Run(context.Background(), c, "Hi! I'm Jim")
	require.NoError(t, err)

	res, err := Run(context.Background(), c, "What is my name?")
	require.NoError(t, err)
	require.True(t, strings.Contains(res, "Jim"), `result does not contain the keyword 'Jim'`)
}
