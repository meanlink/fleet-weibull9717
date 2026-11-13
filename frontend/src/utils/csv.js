/* 旧批次可能没有 start_km 列；缺列不补键，空值按逐件缺省处理。 */
const NUMERIC=['km','failed','start_km'];
export function parseCSV(text){
 const rows=[];let row=[],cell='',quote=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quote&&text[i+1]==='"'){cell+='"';i++;}else quote=!quote;}else if(c===','&&!quote){row.push(cell);cell='';}else if((c==='\n'||c==='\r')&&!quote){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(Boolean))rows.push(row);row=[];cell='';}else cell+=c;}
 if(quote)throw new Error('CSV 引号未闭合');row.push(cell);if(row.some(Boolean))rows.push(row);
 const keys=rows.shift()?.map(k=>k.replace(/^\uFEFF/,'').trim());if(!keys?.length)throw new Error('CSV 为空');
 return rows.map((r,i)=>{if(r.length!==keys.length)throw new Error('CSV 第 '+(i+2)+' 行列数不一致');const record=Object.fromEntries(keys.map((k,j)=>[k,r[j]?.trim()]));for(const key of NUMERIC){if(typeof record[key]==='string'&&record[key]!==''){const n=Number(record[key]);if(Number.isFinite(n))record[key]=n;}}return record;});
}
