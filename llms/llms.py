import os
import re
import statistics
import threading
import time
import queue
import concurrent.futures
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional, Tuple, Type, Union, Dict, Any, Callable

from prettytable import PrettyTable

from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    BedrockAnthropicProvider,
    AI21Provider,
    CohereProvider,
    AlephAlphaProvider,
    HuggingfaceHubProvider,
    GoogleGenAIProvider,
    GoogleVertexAIProvider,
    MistralProvider,
    OllamaProvider,
    DeepSeekProvider,
    GroqProvider,
    RekaProvider,
    TogetherProvider,
    OpenRouterProvider,
)
from .providers.base_provider import BaseProvider
from .results.result import AsyncStreamResult, Result, Results, StreamResult


LOGGER = getLogger(__name__)


@dataclass
class Provider:
    provider: Type[BaseProvider]
    api_key_name: Optional[str] = None
    api_key: Optional[str] = None
    needs_api_key: bool = True
    custom_credential_check: Optional[Callable[[], bool]] = None


def create_provider(
    provider_class: Type[BaseProvider],
    api_key_name: Optional[str] = None,
    needs_api_key: bool = True,
    custom_credential_check: Optional[Callable[[], bool]] = None,
) -> Provider:
   
    return Provider(
        provider_class, 
        api_key_name=api_key_name, 
        needs_api_key=needs_api_key,
        custom_credential_check=custom_credential_check
    )


class LLMS:
    _provider_map: Dict[str, Provider] = {
        "OpenAI": create_provider(OpenAIProvider, "OPENAI_API_KEY"),
        "Anthropic": create_provider(AnthropicProvider, "ANTHROPIC_API_KEY"),
        "BedrockAnthropic": create_provider(
            BedrockAnthropicProvider, 
            needs_api_key=True,
            custom_credential_check=lambda: all([
                os.getenv("AWS_ACCESS_KEY_ID"),
                os.getenv("AWS_SECRET_ACCESS_KEY")
            ])
        ),
        "AI21": create_provider(AI21Provider, "AI21_API_KEY"),
        "Cohere": create_provider(CohereProvider, "COHERE_API_KEY"),
        "AlephAlpha": create_provider(AlephAlphaProvider, "ALEPHALPHA_API_KEY"),
        "HuggingfaceHub": create_provider(
            HuggingfaceHubProvider, "HUGGINFACEHUB_API_KEY"
        ),
        "GoogleGenAI": create_provider(GoogleGenAIProvider, "GOOGLE_API_KEY"),
        "GoogleVertexAI": create_provider(
            GoogleVertexAIProvider, 
            needs_api_key=False,
            custom_credential_check=lambda: bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
        ),
        "Mistral": create_provider(MistralProvider, "MISTRAL_API_KEY"),
        "Ollama": create_provider(OllamaProvider, needs_api_key=False),
        "DeepSeek": create_provider(DeepSeekProvider, "DEEPSEEK_API_KEY"),
        "Groq": create_provider(GroqProvider, "GROQ_API_KEY"),
        "Reka": create_provider(RekaProvider, "REKA_API_KEY"),
        "Together": create_provider(TogetherProvider, "TOGETHER_API_KEY"),
        "OpenRouter": create_provider(OpenRouterProvider, "OPENROUTER_API_KEY"),
    }
    _providers: List[BaseProvider] = []
    _models: List[str] = []

    def __init__(
        self, model: Union[str, List[str], None] = None, **kwargs: Any
    ) -> None:
        """Programmatically load api keys and instantiate providers."""
        self._load_api_keys(kwargs)
        self._set_models(model)
        self._initialize_providers(kwargs)

    def __repr__(self) -> str:
        return f"LLMS({','.join(self._models)})"

    @property
    def n_provider(self) -> int:
        return len(self._providers)

    def list(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "provider": provider.__name__,
                "name": model,
                "cost": cost,
            }
            for provider in [p.provider for p in self._provider_map.values()]
            for model, cost in provider.MODEL_INFO.items()
            if not query
            or (
                query.lower() in model.lower()
                or query.lower() in provider.__name__.lower()
            )
        ]

    def count_tokens(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> Union[int, List[int]]:
        results = [provider.count_tokens(content) for provider in self._providers]
        return results if self.n_provider > 1 else results[0]

    def _process_completion(
        self, prompt: str, is_async: bool, **kwargs: Any
    ) -> Union[Result, Results]:
        async def _async_generate(provider):
            return await provider.acomplete(prompt, **kwargs)

        def _sync_generate(provider):
            return provider.complete(prompt, **kwargs)

        if self.n_provider > 1:
            if is_async:

                async def gather_results():
                    return await asyncio.gather(
                        *[_async_generate(provider) for provider in self._providers]
                    )

                results = asyncio.run(gather_results())
            else:
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(_sync_generate, self._providers))
            return Results(results)
        else:
            provider = self._providers[0]
            return _async_generate(provider) if is_async else _sync_generate(provider)

    def complete(self, prompt: str, **kwargs: Any) -> Union[Result, Results]:
        return self._process_completion(prompt, is_async=False, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> Union[Result, Results]:
        return await self._process_completion(prompt, is_async=True, **kwargs)

    def complete_stream(self, prompt: str, **kwargs: Any) -> StreamResult:
        if self.n_provider > 1:
            raise ValueError("Streaming is possible only with a single model")
        return self._providers[0].complete_stream(prompt, **kwargs)

    async def acomplete_stream(self, prompt: str, **kwargs: Any) -> AsyncStreamResult:
        if self.n_provider > 1:
            raise ValueError("Streaming is possible only with a single model")
        return await self._providers[0].acomplete_stream(prompt, **kwargs)

    def benchmark(
        self,
        problems: Optional[List[Tuple[str, str]]] = None,
        evaluator: Optional[BaseProvider] = None,
        show_outputs: bool = False,
        html: bool = False,
        delay: float = 0,
        reasoning_effort: Optional[str] = None,
        max_tokens: int = 1000,
        thinking: Optional[int] = None,
        temperature: float = 0,
        **kwargs: Any,
    ) -> Union[PrettyTable, str]:
        if not problems:
            problems = [
                (
                    "Write a one paragraph cover letter for a job in a tech company. Make sure to use the word \"the\" exactly twice.",
                    "Correct answer will use the word 'the' exactly twice.",
                ),
                (
                    "write three sentences, each ending with the word 'band'",
                    "Correct answer will have three sentences, and each sentence will have 'band' as the last word",
                ),
                (
                    "what is the capital of finland? if it begins with a letter h, respond 'Oslo' otherwise respond Helsinki.",
                    "Oslo",
                ),
                (
                    "write a sentence about trees with no words beginning with the letter t",
                    "Correct answer will have no words begin with the letter t",
                ),
                (
                    "write 7 numbers between 10 and 110. none of them should begin with 1,5,3,4,2,6,8 or 7",
                    "Correct answer will have 7 numbers and they will be between 90 and 99",
                ),
                (
                    "If a + b + c = 30 and b = 10 and c = 5. Is a = 20? Answer only \"My answer is yes.\" or \"My answer is no.\" or \"My answer is maybe.\"",
                    "My answer is no.",
                ),
                (
                    """given sentence 'today is a sunny day' and instructions 

1. replace words with number of commas equal to the length of the word 

2. if there are three or more commas in the new sentence, replace commas with dots

print the output""",
                    "...... ,, , ...... ...",
                ),
                (
                    'Given the sentence "The cat jumped over the fence twice" write the sentence again adding a number in square brackets after each word corrsepnsing to its poistion in the sentence starting with 1. then add those numbers and write a the sum at the end number with no brackets.',
                    "answer should match exactly this sequence: The [1] cat [2] jumped [3] over [4] the [5] fence [6] twice [7] 28",
                ),
                (
                    "A glass door has 'push' written on it in mirror writing. Should you push or pull it and why?",
                    "pull",
                ),
                (
                    'Given the string: "A# B# #B A# A# #B #B A# A# #B A# A#" Could you check for any instances of "A# #B" and replace them with "B# #A"? print only the answer',
                    "B# B# #A B# B# #A #A B# B# #A B# B#",
                ),
                (
                    "Kevin currently has 8 apples. He ate 3 apples yesterday. How many apples does Kevin have now?",
                    "8",
                ),
                (
                    'What is the largest land animal? If that animal has wings, answer "The African Elephant". Otherwise, answer "The Mouse". Do not provide any explanation for your choice.',
                    "The Mouse",
                ),
                (
                    "Oliver picks 34 kiwis on Friday. Then he picks 58 kiwis on Saturday. On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average. How many kiwis does Oliver have?",
                    "150",
                ),
                (
                    "In my kitchen there's a table with a cup with a ball inside. I moved the cup to my bed in my bedroom and turned the cup upside down. I grabbed the cup again and moved to the main room. Where's the ball now?",
                    "on the bed in the bedroom",
                ),
                (
                    """#define _POSIX_SOURCE
#include                                <time.h>
#include                               <stdio.h>
#define                             extern/* ioccc*/
#define                           condition 22+d+e> 31
#define                         declarator main ( void )
#define                       keyword ae(t,d+e-9); ae(m,4)
#define                     false ae (t, d+ e+ 22); ae (m,3)
#define                   syntax_error(n); } else if (J < n) {
#define                 relational_expression(i, o, ccc) i o ccc
#define               errno translation_unit/*Apollo 13*/int errno
#define             iterative_statement  for  (  expressions ) block
#define             translation_unit  declarator  compound_statement
#define             ae(unary_expression,ae)     unary_expression= ae
#define             declaration char T[16];     struct tm ae(s,{ 0})
#define             if_part {macrolist list     ae(b,J%4); type_name
#define             tokens if (t==25&&d==28     &&a > 10) ae (t, 18)
#define             selection_statement(a,b,c) if(a){b; } else{ c; }
#define             storage_class ae(e,(2*b+4*c+6*d+N)%7); statement
#define             statement_list iterative_statement return'^'^'^'
#define            macro(x,y,cast) ae(M,x); ae(N,y)syntax_error(cast)
#define           block { if(relational_expression(J,<,1700))if_part;}
#define          declaration_list int J,a,b,c,d, e,t,m,M,N; declaration
#define         true keyword; selection_statement(t==26,ae(t,19),tokens)
#define        compound_statement { declaration_list ; statement_list ; }
#define       expressions ae(J,1582); relational_expression(J,<,2200); ++J
#define      list macro (24, 5, 2200) ae(M, 24); ae(N, 6); } ae(a, J % 19);
#define     type_name ae (c, J % 7);ae (d, (19 * a + M) % 30); storage_class
#define    statement  selection_statement ( condition,true,false)pptoken cast
#define   macrolist macro(22,2,1800)macro     (23, 3,1900) macro (23, 4, 2100)
#define  pptoken ae(s.tm_year,J-1900);ae       (s.tm_mon,m-1); ae(s.tm_mday,t);
#define cast (void)strftime(T,sizeof T,         "%a %b %d %Y",&s),(void)puts(T);
#include                               <errno.h>

what does this do, in one sentence?""",
                    "Calculate Easter dates within the Gregorian Calendar.",
                ),
                (
                    """#include <stdio.h> 

#define N(a)       "%"#a"$hhn"
#define O(a,b)     "%10$"#a"d"N(b)
#define U          "%10$.*37$d"
#define G(a)       "%"#a"$s"
#define H(a,b)     G(a)G(b)
#define T(a)       a a 
#define s(a)       T(a)T(a)
#define A(a)       s(a)T(a)a
#define n(a)       A(a)a
#define D(a)       n(a)A(a)
#define C(a)       D(a)a
#define R          C(C(N(12)G(12)))
#define o(a,b,c)   C(H(a,a))D(G(a))C(H(b,b)G(b))n(G(b))O(32,c)R
#define SS         O(78,55)R "\n\033[2J\n%26$s";
#define E(a,b,c,d) H(a,b)G(c)O(253,11)R G(11)O(255,11)R H(11,d)N(d)O(253,35)R
#define S(a,b)     O(254,11)H(a,b)N(68)R G(68)O(255,68)N(12)H(12,68)G(67)N(67)

char* fmt = O(10,39)N(40)N(41)N(42)N(43)N(66)N(69)N(24)O(22,65)O(5,70)O(8,44)N(
            45)N(46)N    (47)N(48)N(    49)N( 50)N(     51)N(52)N(53    )O( 28,
            54)O(5,        55) O(2,    56)O(3,57)O(      4,58 )O(13,    73)O(4,
            71 )N(   72)O   (20,59    )N(60)N(61)N(       62)N (63)N    (64)R R
            E(1,2,   3,13   )E(4,    5,6,13)E(7,8,9        ,13)E(1,4    ,7,13)E
            (2,5,8,        13)E(    3,6,9,13)E(1,5,         9,13)E(3    ,5,7,13
            )E(14,15,    16,23)    E(17,18,19,23)E(          20, 21,    22,23)E
            (14,17,20,23)E(15,    18,21,23)E(16,19,    22     ,23)E(    14, 18,
            22,23)E(16,18,20,    23)R U O(255 ,38)R    G (     38)O(    255,36)
            R H(13,23)O(255,    11)R H(11,36) O(254    ,36)     R G(    36 ) O(
            255,36)R S(1,14    )S(2,15)S(3, 16)S(4,    17 )S     (5,    18)S(6,
            19)S(7,20)S(8,    21)S(9    ,22)H(13,23    )H(36,     67    )N(11)R
            G(11)""O(255,    25 )R        s(C(G(11)    ))n (G(          11) )G(
            11)N(54)R C(    "aa")   s(A(   G(25)))T    (G(25))N         (69)R o
            (14,1,26)o(    15, 2,   27)o   (16,3,28    )o( 17,4,        29)o(18
            ,5,30)o(19    ,6,31)o(        20,7,32)o    (21,8,33)o       (22 ,9,
            34)n(C(U)    )N( 68)R H(    36,13)G(23)    N(11)R C(D(      G(11)))
            D(G(11))G(68)N(68)R G(68)O(49,35)R H(13,23)G(67)N(11)R C(H(11,11)G(
            11))A(G(11))C(H(36,36)G(36))s(G(36))O(32,58)R C(D(G(36)))A(G(36))SS

#define arg d+6,d+8,d+10,d+12,d+14,d+16,d+18,d+20,d+22,0,d+46,d+52,d+48,d+24,d\
            +26,d+28,d+30,d+32,d+34,d+36,d+38,d+40,d+50,(scanf(d+126,d+4),d+(6\
            -2)+18*(1-d[2]%2)+d[4]*2),d,d+66,d+68,d+70, d+78,d+80,d+82,d+90,d+\
            92,d+94,d+97,d+54,d[2],d+2,d+71,d+77,d+83,d+89,d+95,d+72,d+73,d+74\
            ,d+75,d+76,d+84,d+85,d+86,d+87,d+88,d+100,d+101,d+96,d+102,d+99,d+\
            67,d+69,d+79,d+81,d+91,d+93,d+98,d+103,d+58,d+60,d+98,d+126,d+127,\
            d+128,d+129

char d[538] = {1,0,10,0,10};

int main() {
    while(*d) printf(fmt, arg);
}

what does this program do in one sentence?""",
                    "it is an obfuscated implementation of a Tic Tac Toe game",
                ),
                (
                    """#include <stdio.h>
#define  f(f,g){z e=0;for(;e<f;e++)g;}
#define  i(f,g)static z f(z a){return g;}
#define  j(f,g)static void f(z*a,z*b,z*c){g}
#define  h(f,g)static z f(z a,z b,z c){return g;}
#define  g(f,g,h,i,j)static z f(z b){z a=g,c=h;for(;i)a=j;return a;}
typedef unsigned char y;typedef unsigned long long z;extern y*w;static z b(z a,z b){return a>>b|a<<(64-b);}i(_,
(a>>6)^b(a,61)^b(a,19))i(_a,b(a,39)^b(a,28)^b(a,34))h(x,((a^b)&c)^(a&b))i(u,b(a,41)^b(a,18)^b(a,14))h(t,(((((3*(a*c+b*b)>>9)+(3*
b*c>>32))*a>>21)+(3*a*a*b>>6)+((b>>4)*(b>>4)*b>>46))>>18)+a*a*a)h(m,t((b<<16)|(c>>48),(c>>24)%(1<<24),c%(1<<24))>>48<a)h(s,(a&b)
^(~a&c))i(r,b(a,1)^b(a,8)^(a>>7))g(o,0,0,c<8;c++,a*256+w[b*8+c])g(d,0,0,c<13;c++,a*31+w[b*13+c]-96)g(p,0,4,c;c/=2,a|c*m(b,a|c,a)
)g(q,0,1ull<<63,c;c/=2,a|c*m(b,p(b),a|c))g(v,b>1,2,c<b;c++,a&&b%c)g(l,b?l(b-1)+1:2,a,!v(c);c++,c+1)j(n,z d=a[7]+u(a[4])+s(a[4],a
[5],a[6])+q(l(*b))+c[*b%16];f(8,a[7-e]=e-3?e-7?a[6-e]:d+_a(a[0])+x(a[1],a[2],a[3]):d+a[3])f(16*(*b%16>14),c[e]+=c[(e+9)%16]+r(c[
(e+1)%16])+_(c[(e+14)%16])))j(k,f(8,b[e]=a[e])f(80,n(a,&e,c))f(8,a[e]+=b[e]))int main(){z a[8],b[8],c[16];f(8,a[e]=d(e))f(16,c[e
]=e-15?o(e):d(8))k(a,b,c);f(16,c[e]=e?e-15?0:11264:1ull<<63)k(a,b,c);f(8,printf("%016llx%s",a[e],e-7?"":"\n"))return!w;}y*w=(y*)
"crsmyiajqhwy{unwa|hjoi`hlxhpxrzb~edko~rtr~ileqyjk`znqgsuitvgqnfdfa||wedvnmhozkpokootqzcexeld~oibqzpcsuw{ib{x`m`hsa`jmn}wcfzpb";

what does this program do, in one sentence?""",
                    "This program prints its own SHA-512 hash",
                ),
                ("How many w's are in strawberry?", "1"),
                (
                    "Use  g to substitute c, m to substitute p, a to substitute e, o to substitute h and n to substitute a\
how to spell cheap under this rule?",
                    "goanm",
                ),
                (
                    "two workers paint the fence in 8 hours. how long will it take one worker paint the same fence if they are injured and need to take a 30 min break after every hour of work?",
                    "23.5 hours",
                ),
                (
                    "Alan, Bob, Colin, Dave and Emily are standing in a circle. Alan is on Bob's immediate left. Bob is on Colin's immediate left. Colin is on Dave's immediate left. Dave is on Emily's immediate left. Who is on Alan's immediate right?",
                    "Bob",
                ),
                ("-2-2-2-2-2-2*-2*-2-2/-2=", "-17"),
                ('what is the 13th letter of the word "supralapsarian"', "a"),
                (
                    "A loaf of sourdough at the cafe costs $8. Muffins cost $3 each. If we purchase 10 loaves of sourdough and 10 muffins, how much more do the sourdough loaves cost compared to the muffins, if we plan to donate 3 loaves of sourdough and 2 muffins from this purchase?",
                    "$50",
                ),
                (
                    """Liam wants to buy some school supplies. He buys 24 erasers that now cost $6.75 each, 10 notebooks that now cost $11.0 each, and a ream of bond paper that now costs $9. How much should Liam pay now, taking into account that due to inflation, prices were 10% cheaper last year!
                    """,
                    "$281",
                ),
                (
                    "what is the minimum number of tests in worse scenario to test 8 identical looking batteries of which 4 are good and 4 bad in flashlight that takes 2 batteries, to make sure two good batteries are inserted (counting last insert as one try) ",
                    "6",
                ),
                (
                    "Sort the following list into alphabetical order. apple, code, banana, gun, orange, grape, box, cube. Separate items with as many asterisk characters (*) as the length of previous word",
                    "answer should exactly match this sequence: apple*****banana******box***code****cube****grape*****gun***orange",
                ),
                (
                    'Vlad\'s uncle can still beat him in sprinting although he is 30 years younger. who is "he" referring to?',
                    "Vlad",
                ),
                (
                    "A farmer and a sheep are standing on one side of a river. There is a boat with enough room for one human and one animal. what is the fewest number of trips for farmer to get across the river with the sheep?",
                    "one",
                ),
                (
                    "The least common multiple of a positive integer n and 18 is 180, and the greatest common divisor of n and 45 is 15. What is the sum of the digits of n?",
                    "n = 60 thus the answer is 6",
                ),
                (
                    """section .data
    a dd 0
    b dd 0

section .text
    global _start

_start:
    mov eax, [a]
    add eax, [b]
    mov [a], eax
    mov eax, [a]
    sub eax, [b]
    mov [b], eax
    mov eax, [a]
    sub eax, [b]
    mov [a], eax

    mov eax, 60
    xor edi, edi
    syscall

what does this do, in one sentence?""",
                    "swaps the values of two variables 'a' and 'b' using without using a temporary variable",
                ),
                (
                    """The sky was a brilliant shade of blue, dotted with fluffy white clouds.
                In the above sentence add the word 'green' after 'blue' and 'black' after 'clouds'. then remove words 'shade' and 'clouds'. then move the word 'sky' forward three words. Substitute "brilliant" with "vibrant". Replace "dotted" with "adorned". Move "was a" to the end of the sentence. what is left?""",
                    "Answer should match this exactly: The vibrant sky of blue green, adorned with fluffy white black was a.",
                ),
                (
                    """Begin with these sentence:
"The quick brown fox jumps over the lazy dog. A wizard's job is to vex chumps quickly in fog."
Now, follow these instructions:

Remove all words containing the letter 'i'.""",
                    "Answer should match this exactly: The brown fox jumps over the lazy dog. A job to vex chumps fog.",
                ),
                (
                    """Begin with these sentence:
"The quick brown fox jumps over the lazy dog. A wizard's job is to vex chumps quickly in fog."
Now, follow these instructions:
- Insert the word "cybernetic" after every word ending in 's'.""",
                    "Answer should match this exactly: The quick brown fox jumps cybernetic over the lazy dog. A wizard's cybernetic job is cybernetic to vex chumps cybernetic quickly in fog.",
                ),
                (
                    "what square is the black king on in this chess position: 1Bb3BN/R2Pk2r/1Q5B/4q2R/2bN4/4Q1BK/1p6/1bq1R1rb w - - 0 1",
                    "e7",
                ),
                (
                    "In a room there are only three sisters. Anna is reading a book. Alice is playing chess with someone. What is the third sister, Amanda doing?",
                    "Playing chess with Alice",
                ),
                ("is 9.11 bigger number than 9.9?", "no, it is not"),
                (
                    "You have six horses and want to race them to see which is fastest. How many races would you need to do this?",
                    "one",
                ),
                ("I do not not not like eggs. Do I like eggs?", "No"),
                (
                    "John has two brothers - called Snap and Crackle. The three children's names are: Snap, Crackle and _.",
                    "John",
                ),
                (
                    "How many boxes do I have if I have three boxes with one box inside each, and one box inside them?",
                    "9",
                ),
                (
                    "Given a QWERTY keyboard layout, if HEART goes to JRSTY, what does QWERTY go to?",
                    "WERTYU",
                ),
                (
                    "An arrow points up. We rotate it 90 degrees to the clockwise, mirror it along its flat end, and rotate it another 90 degrees clockwise. Which direction is it pointing?",
                    "up",
                ),
                (
                    """1. Start with the word "CIPHER".
2. Count the number of letters in "CIPHER". Add 1 to this number.
3. Take the letter in "ALPHABET" at the position of the number you got in step 2 and remove it.
Print the output""",
                    "Answer should match this exactly: ALPHABT",
                ),
                (
                    "Current flight information (the following flights are one-way only, and all the flights available are included below):\n\
There is a flight from city G to city B\n\
There is a flight from city H to city K\n\
There is a flight from city L to city M\n\
There is a flight from city F to city H\n\
There is a flight from city G to city J\n\
There is a flight from city B to city I\n\
There is a flight from city L to city A\n\
There is a flight from city H to city N\n\
There is a flight from city B to city D\n\
There is a flight from city J to city C\n\
Question: Is there a series of flights that goes from city F to city I?",
                    "No, there is no series of flights from F to I",
                ),
                (
                    "Bob (a boy) has 3 sisters. Each sister has 2 brothers. How many brothers does Bob have?",
                    "1",
                ),
                (
                    "Imagine there is a circular pond in an oasis, with two trees at the edge of the pond, on opposite sides. Bob sets up a hammock by hanging it between the two trees. He gets into the hammock and falls asleep. If he were to roll over in his sleep and fall out of the hammock, where would he fall?",
                    "water, in the center of the pond",
                ),
                (
                    "Beth places four whole ice cubes in a frying pan at the start of the first minute, then five at the start of the second minute and some more at the start of the third minute, but none in the fourth minute. If the average number of ice cubes per minute placed in the pan while it was frying a crispy egg was five, how many whole ice cubes can be found in the pan at the end of the third minute?",
                    "0",
                ),
                (
                    "A juggler throws a solid blue ball a meter in the air and then a solid purple ball (of the same size) two meters in the air. She then climbs to the top of a tall ladder carefully, balancing a yellow balloon on her head. Where is the purple ball most likely now, in relation to the blue ball?",
                    "at the same height as the blue ball (both on the ground)",
                ),
                (
                    "Jeff, Jo and Jim are in a 200m men's race, starting from the same position. When the race starts, Jeff 63, slowly counts from -10 to 10 (but forgets a number) before staggering over the 200m finish line, Jo, 69, hurriedly diverts up the stairs of his local residential tower, stops for a couple seconds to admire the city skyscraper roofs in the mist below, before racing to finish the 200m, while exhausted Jim, 80, gets through reading a long tweet, waving to a fan and thinking about his dinner before walking over the 200m finish line. Who likely finished last?",
                    "Jo likely finished last",
                ),
                (
                    "There are two sisters, Amy who always speaks mistruths and Sam who always lies. You don't know which is which. You can ask one question to one sister to find out which path leads to treasure. Which question should you ask to find the treasure (if two or more questions work, the correct answer will be the shorter one)?",
                    "What path leads to the treasure?",
                ),
                (
                    "Agatha makes a stack of 5 cold, fresh single-slice ham sandwiches (with no sauces or condiments) in Room A, then immediately uses duct tape to stick the top surface of the uppermost sandwich to the bottom of her walking stick. She then walks to Room B, with her walking stick, so how many whole sandwiches are there now, in each room?",
                    "4 whole sandwiches in room A, 0 whole sandwiches in Room B",
                ),
                (
                    "A luxury sports-car is traveling north at 30km/h over a roadbridge, 250m long, which runs over a river that is flowing at 5km/h eastward. The wind is blowing at 1km/h westward, slow enough not to bother the pedestrians snapping photos of the car from both sides of the roadbridge as the car passes. A glove was stored in the trunk of the car, but slips out of a hole and drops out when the car is half-way over the bridge. Assume the car continues in the same direction at the same speed, and the wind and river continue to move as stated. 1 hour later, the water-proof glove is (relative to the center of the bridge) approximately northward, eastward, north-easterly or north-westerly?",
                    "northward",
                ),
                (
                    "Hoping to break their current losing streak the Cowboys played on home ground for an Interconference duel with the Jaguars. In the first quarter the Cowboys took the lead as kicker David Buehler hit a 34-yard field goal. But they fell behind with QB David Garrard getting a 10-yard TD pass to WR Mike Sims-Walker. In the second quarter, the Cowboys struggled further with Garrard finding TE Marcedes Lewis on a 42-yard TD pass, then in the third quarter he found WR Mike Thomas on a 15-yard TD pass, and then he found Lewis again on a 9-yard TD pass. The Cowboys responded in the 4th quarter with RB Marion Barber getting a 1-yard TD run. But the Jaguars scored again with Garrard scrambling 2 yards to the endzone for a touchdown. The Cowboys replied with QB Jon Kitna making an 8-yard TD pass to TE Jason Witten.   What was the shortest TD pass of the third quarter?",
                    "9-yard TD pass",
                ),
                (
                    "According to CBS, in 2001 the ethnic makeup of the city was 99.8% Jewish and other non-Arab, without significant Arab population. See Population groups in Israel. According to CBS, in 2001 there were 23,700 males and 24,900 females. The population of the city was spread out with 31.4% 19 years of age or younger, 15.7% between 20 and 29, 18.5% between 30 and 44, 18.3% from 45 to 59, 4.1% from 60 to 64, and 11.9% 65 years of age or older. The population growth rate in 2001 was 0.8%. How many more people (in percentage) were in the 2 biggest age groups combined compared to the 2 smallest age groups combined?",
                    "33.9",
                ),
                (
                    """For the period 2010-14, the estimated median income for a household in the town was $94,063, and the median income for a family was $129,000. Male full-time workers had a median income of $87,550 versus $53,141 for females. The per capita income for the town was $34,140. About 2.0% of families and 12.0% of the population were below the poverty line, including 3.4% of those under age 18 and 4.8% of those age 65 or over.how many percent of the population aged 65 or over were not below the poverty line?""",
                    "88",
                ),
                (
                    "What is the largest even integer that cannot be written as the sum of two odd composite numbers?",
                    "38",
                ),
                (
                    "Jenny and Kenny are walking in the same direction, Kenny at 3 feet per second and Jenny at 1 foot per second, on parallel paths that are 200 feet apart. A tall circular building 100 feet in diameter is centered midway between the paths. At the instant when the building first blocks the line of sight between Jenny and Kenny, they are 200 feet apart. Let $t\\,$ be the amount of time, in seconds, before Jenny and Kenny can see each other again. If $t\\,$ is written as a fraction in lowest terms, what is the sum of the numerator and denominator?",
                    "163",
                ),
                (
                    "Find the smallest prime that is the fifth term of an increasing arithmetic sequence, all four preceding terms also being prime.",
                    "29",
                ),
                (
                    "Abe can paint the room in 15 hours, Bea can paint 50 percent faster than Abe, and Coe can paint twice as fast as Abe. Abe begins to paint the room and works alone for the first hour and a half. Then Bea joins Abe, and they work together until half the room is painted. Then Coe joins Abe and Bea, and they work together until the entire room is painted. Find the number of minutes after Abe begins for the three of them to finish painting the room.",
                    "334",
                ),
                ("In the fibonnaci sequence starting with 10, then seven unknown numbers in sequence and then 11, find the number in the sequence after 10. All numbers in this sequence adhere to fibonnaci sequence rules. Output solution as a fraction.", "-17/3"),
                ("i have three balls next to each other; yellow then to the right blue then to the right red. i take two outmost balls and swap them then take blue and swap with yellow, then take two outmost and swap them, then swap  blue and yellow again. output just final orders of the balls and no other text", "yellow, blue, red"),
                
            ]

        def evaluate_answers(
            evaluator, query_answer_pairs: List[Tuple[str, str, str]]
        ) -> List[int]:
            system = """You are an evaluator for an AI system. Your task is to determine whether the AI's answer matches the correct answer. You will be given two inputs: the AI's answer and the correct answer. Your job is to compare these and output a binary score: 1 if the AI's answer is correct, and 0 if it is not.

        To evaluate the AI's performance:
        1. Carefully compare the AI's answer to the correct answer.
        2. Consider the following:
           - Does the AI's answer convey the same meaning as the correct answer?
           - Are there any significant discrepancies or omissions in the AI's answer?
           - If there are minor differences in wording but the core information is the same, consider it correct.

        After your evaluation, provide your assessment in the following format:
        <evaluation>
        [Your reasoning for the score]
        </evaluation>
        <score>[0 or 1]</score>

        Remember, output only 0 (not correct) or 1 (correct) as the final score. Do not include any additional explanation or text outside of the specified tags."""

            scores = []
            for i, (query, correct_answer, ai_answer) in enumerate(
                query_answer_pairs, start=1
            ):
                prompt = f"""Here is the AI's answer:
        <ai_answer>
        {ai_answer}
        </ai_answer>
        Here is the correct answer:
        <correct_answer>
        {correct_answer}
        </correct_answer>"""

                try:
                    evaluator_result = evaluator.complete(
                        prompt, system_message=system
                    ).text
                except Exception as e:
                    LOGGER.error(f"Evaluation call failed for query {i}: {e}")
                    scores.append(0)          # fall-back score
                    continue

                # Extract score (allow whitespace) – expect 0 or 1, else default to 0
                score_match = re.search(r"<score>\s*([01])\s*</score>", evaluator_result)
                # NEW fallback – look for a bare 0/1 anywhere if the tag-based search failed
                if not score_match:
                    score_match = re.search(r"\b([01])\b", evaluator_result.strip())

                if score_match:
                    scores.append(int(score_match.group(1)))
                else:
                    LOGGER.warning(
                        f"Could not extract score for query {i}. "
                        f"Raw evaluator output: {evaluator_result[:200]}..."
                    )
                    scores.append(0)

            return scores

        model_results = {}

        def process_prompt(model, prompt, index, evaluator, evaluation_queue, **kwargs):
            try:
                print(model, index)  # , prompt[0])
                # Prepare kwargs for complete call
                complete_kwargs = {
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    **kwargs
                }
                
                # Only add optional parameters if they have values
                if reasoning_effort is not None:
                    complete_kwargs['reasoning_effort'] = reasoning_effort
                if thinking is not None:
                    complete_kwargs['thinking'] = thinking
                    
                result = model.complete(prompt[0], **complete_kwargs)
                if delay > 0:
                    time.sleep(delay)
                output_data = {
                    "text": result.text,
                    "tokens": result.meta["tokens_completion"],
                    "latency": result.meta["latency"],
                    "cost": result.meta["cost"],
                    "prompt_index": index,
                }
            except Exception as e:
                print(f"Error with {model}: {str(e)}")
                return None

            if evaluator:
                def _eval_target():
                    try:
                        score = evaluate_answers(
                            evaluator, [(prompt[0], prompt[1], result.text)]
                        )[0]
                    except Exception as e:
                        LOGGER.error(f"Evaluation error for prompt index {index}: {e}")
                        score = 0
                    evaluation_queue.put((index, score))

                evaluation_thread = threading.Thread(target=_eval_target)
                evaluation_thread.start()
                output_data["evaluation_thread"] = evaluation_thread

            return output_data

        def process_prompts_sequentially(model, prompts, evaluator, **kwargs):
            results = []
            evaluation_queue = queue.Queue()
            evaluation_threads = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(
                        process_prompt,
                        model,
                        prompt,
                        index,
                        evaluator,
                        evaluation_queue,
                        **kwargs,
                    )
                    for index, prompt in enumerate(prompts)
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        if evaluator and "evaluation_thread" in result:
                            evaluation_threads.append(result.get("evaluation_thread"))
            return model, results, evaluation_queue, evaluation_threads

        # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_prompts_sequentially, model, problems, evaluator, **kwargs
                )
                for model in self._providers
            ]

            for future in as_completed(futures):
                try:
                    (
                        model,
                        outputs,
                        evaluation_queue,
                        evaluation_threads,
                    ) = future.result()
                    if not outputs:  # Skip if no successful outputs
                        continue

                    if outputs:  # Only process if we have valid outputs
                        model_results[model] = {
                            "outputs": outputs,
                            "total_latency": 0,
                            "total_cost": 0,
                            "evaluation": [None] * len(outputs),
                        }

                        for output_data in outputs:
                            if output_data:  # Check if output_data is not None
                                model_results[model]["total_latency"] += output_data["latency"]
                                model_results[model]["total_cost"] += output_data["cost"]

                        if evaluator and evaluation_threads:
                            # Wait for all evaluation threads to complete
                            for thread in evaluation_threads:
                                if thread:  # Check if thread exists
                                    thread.join()

                            # Process all evaluation results
                            while not evaluation_queue.empty():
                                index, evaluation = evaluation_queue.get()
                                model_results[model]["evaluation"][index] = evaluation
                except Exception as e:
                    print(f"Error processing results: {str(e)}")
                    # Don't add failed models to results
                    continue

        for model in model_results:
            outputs = model_results[model]["outputs"]
            model_results[model]["median_latency"] = statistics.median(
                [output["latency"] for output in outputs]
            )

            total_tokens = sum([output["tokens"] for output in outputs])
            total_latency = model_results[model]["total_latency"]
            model_results[model]["aggregated_speed"] = total_tokens / total_latency

        if evaluator:
            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"]
                * sum(filter(None, model_results[x]["evaluation"])),
                reverse=True,
            )
        else:
            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"],
                reverse=True,
            )

        headers = [
            "Model",
            "Output",
            "Tokens",
            "Cost ($)",
            "Latency (s)",
            "Speed (tokens/sec)",
            "Evaluation",
        ]

        if not show_outputs:
            headers.remove("Output")

        if not evaluator:
            headers.remove("Evaluation")

        table = PrettyTable(headers)

        for model in sorted_models:
            model_data = model_results[model]

            total_tokens = 0
            total_score = 0
            valid_evaluations = 0
            for index, output_data in enumerate(model_data["outputs"]):
                total_tokens += output_data["tokens"]
                if evaluator and model_results[model]["evaluation"][index] is not None:
                    total_score += model_results[model]["evaluation"][index]
                    valid_evaluations += 1
                row_data = [
                    str(model),
                    output_data["text"],
                    output_data["tokens"],
                    f'{output_data["cost"]:.5f}',
                    f'{output_data["latency"]:.2f}',
                    f'{output_data["tokens"]/output_data["latency"]:.2f}',
                ]
                if not show_outputs:
                    row_data.remove(output_data["text"])
                if evaluator:
                    row_data.append(model_results[model]["evaluation"][index])
                table.add_row(row_data)

            if show_outputs:
                row_data = [
                    str(model),
                    "",
                    f"{total_tokens}",
                    f"{model_data['total_cost']:.5f}",
                    f"{model_data['median_latency']:.2f}",
                    f"{total_tokens/model_data['total_latency']:.2f}",
                ]

            else:
                row_data = [
                    str(model),
                    f"{total_tokens}",
                    f"{model_data['total_cost']:.5f}",
                    f"{model_data['median_latency']:.2f}",
                    f"{total_tokens/model_data['total_latency']:.2f}",
                ]
            if evaluator:
                if valid_evaluations > 0:
                    acc = 100 * total_score / valid_evaluations
                    row_data.append(f"{acc:.2f}%")
                else:
                    row_data.append("N/A")

            table.add_row(row_data)

        # Track easiest and hardest questions
        easiest_questions = []
        hardest_questions = []
        for i, problem in enumerate(problems):
            valid_results = []
            for model in model_results:
                try:
                    if len(model_results[model]["evaluation"]) > i:
                        eval_result = model_results[model]["evaluation"][i]
                        if eval_result is not None:
                            valid_results.append(eval_result)
                except (IndexError, KeyError):
                    continue
                
            if valid_results:  # Only evaluate if we have valid results
                all_correct = all(result == 1 for result in valid_results)
                all_incorrect = all(result == 0 for result in valid_results)

                if all_correct:
                    easiest_questions.append((i, problem[0]))
                elif all_incorrect:
                    hardest_questions.append((i, problem[0]))

        # Create a new table for easiest and hardest questions
        questions_table = PrettyTable(["Category", "Index", "Question"])
        questions_table.align["Question"] = "l"  # Left-align the Question column

        for index, question in easiest_questions:
            questions_table.add_row(
                [
                    "Easiest",
                    index,
                    question[:100] + ("..." if len(question) > 100 else ""),
                ]
            )

        for index, question in hardest_questions:
            questions_table.add_row(
                [
                    "Hardest",
                    index,
                    question[:100] + ("..." if len(question) > 100 else ""),
                ]
            )

        # Return both tables
        return table, questions_table

        if not html:
            return table, questions_table
        else:
            return table.get_html_string(), questions_table.get_html_string()

    def _load_api_keys(self, kwargs: Dict[str, Any]) -> None:
        self._provider_map = {
            name: Provider(
                provider=provider.provider,
                api_key_name=provider.api_key_name,
                api_key=kwargs.pop(provider.api_key_name.lower(), None)
                or os.getenv(provider.api_key_name) if provider.api_key_name else None,
                needs_api_key=provider.needs_api_key,
            )
            for name, provider in self._provider_map.items()
        }

    def _set_models(self, model: Optional[Union[str, List[str]]]) -> None:
        default_model = os.getenv("LLMS_DEFAULT_MODEL") or "gpt-3.5-turbo"
        self._models = (
            [default_model]
            if model is None
            else ([model] if isinstance(model, str) else model)
        )

    def _validate_model(self, single_model: str, provider: Provider) -> bool:
        return (
            single_model in provider.provider.MODEL_INFO
            and (provider.api_key or not provider.needs_api_key)
        )

    def _initialize_providers(self, kwargs: Dict[str, Any]) -> None:
      
        self._providers = [
            provider.provider(model=single_model, **({**kwargs, 'api_key': provider.api_key} if provider.needs_api_key else kwargs))
            for single_model in self._models
            for provider in self._provider_map.values()
            if self._validate_model(single_model, provider)
        ]

        if not self._providers:
            raise ValueError("No valid providers found for the specified models")

        for provider in self._providers:
            LOGGER.info(
                f"Initialized {provider.model} with {provider.__class__.__name__}"
            )
