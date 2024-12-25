import { Client, TokenizationType } from './client.mjs'

const client = new Client()

try {
    const token = await client.tokenize({ data: 'test@gmail.com', type: TokenizationType.Email })
    console.log(`token: ${token}`)

    const data = await client.detokenize({ data: token, type: TokenizationType.Email })
    console.log(`data: ${data}`)
} catch (err) {
    console.log(err.message)
    process.exit(1)
}