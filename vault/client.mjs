import axios from 'axios'
import * as https from 'node:https'
import * as fs from 'node:fs'

export const TokenizationType = {
    Int32: 1,
    Int64: 2,
    String: 3,
    Bytes: 4,
    Email: 5,
}

export class Client {
    #httpClient = axios.create({
        baseURL: 'https://localhost:9595',
        httpsAgent: new https.Agent({
            rejectUnauthorized: true,
            cert: fs.readFileSync('./ssl/acra-client/acra-client.crt'),
            key: fs.readFileSync('./ssl/acra-client/acra-client.key'),
            ca: fs.readFileSync('./ssl/ca/example.cossacklabs.com.crt'),
        }),
    })

    async tokenize({ data, type = TokenizationType.String } = {}) {
        const res = await this.#httpClient.post('/v2/tokenize', {
            zone_id: '',
            data,
            type,
        })
        return res.data?.data
    }

    async detokenize({ data, type = TokenizationType.String } = {}) {
        const res = await this.#httpClient.post('/v2/detokenize', {
            zone_id: '',
            data,
            type,
        })
        return res.data?.data
    }
}